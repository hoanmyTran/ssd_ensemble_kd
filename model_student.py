import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from transformers import Wav2Vec2Model, UniSpeechSatModel, WavLMModel
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import SSL_Interface
import SSL_Interface.configs
import SSL_Interface.interfaces
from pooling import MultiHeadAttentionPooling

from custom_logger import get_logger
logger = get_logger(__name__)
def compute_det_curve(target_scores, nontarget_scores):
    """ frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, score of target (or positive, bonafide) trials
      nontarget_scores: np.array, score of non-target (or negative, spoofed) trials
      
    output
    ------
      frr:         np.array,  false rejection rates measured at multiple thresholds
      far:         np.array,  false acceptance rates measured at multiple thresholds
      thresholds:  np.array,  thresholds used to compute frr and far

    frr, far, thresholds have same shape = len(target_scores) + len(nontarget_scores) + 1
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    # false rejection rates
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    # false acceptance rates
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    # Thresholds are the sorted scores
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ eer, eer_threshold = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, score of target (or positive, bonafide) trials
      nontarget_scores: np.array, score of non-target (or negative, spoofed) trials
      
    output
    ------
      eer:              scalar,  value of EER
      eer_threshold:    scalar,  value of threshold corresponding to EER
    """

    frr, far, thresholds = compute_det_curve(
        np.array(target_scores).astype(np.longdouble),
        np.array(nontarget_scores).astype(np.longdouble),
    )
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

class PKTCosSim(nn.Module):
	'''
	Learning Deep Representations with Probabilistic Knowledge Transfer
	http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf
	'''
	def __init__(self):
		super(PKTCosSim, self).__init__()

	def forward(self, feat_s, feat_t, eps=1e-6):
        # Normalize each vector by its norm
		feat_s_norm = torch.sqrt(torch.sum(feat_s ** 2, dim=1, keepdim=True))
		feat_s = feat_s / (feat_s_norm + eps)
		feat_s[feat_s != feat_s] = 0

		feat_t_norm = torch.sqrt(torch.sum(feat_t ** 2, dim=1, keepdim=True))
		feat_t = feat_t / (feat_t_norm + eps)
		feat_t[feat_t != feat_t] = 0

		# Calculate the cosine similarity
		feat_s_cos_sim = torch.mm(feat_s, feat_s.transpose(0, 1))
		feat_t_cos_sim = torch.mm(feat_t, feat_t.transpose(0, 1))

		# Scale cosine similarity to [0,1]
		feat_s_cos_sim = (feat_s_cos_sim + 1.0) / 2.0
		feat_t_cos_sim = (feat_t_cos_sim + 1.0) / 2.0

		# Transform them into probabilities
		feat_s_cond_prob = feat_s_cos_sim / torch.sum(feat_s_cos_sim, dim=1, keepdim=True)
		feat_t_cond_prob = feat_t_cos_sim / torch.sum(feat_t_cos_sim, dim=1, keepdim=True)

		# Calculate the KL-divergence
		loss = torch.mean(feat_t_cond_prob * torch.log((feat_t_cond_prob + eps) / (feat_s_cond_prob + eps)))

		return loss
    
class Logits(nn.Module):
	'''
	Do Deep Nets Really Need to be Deep?
	http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
	'''
	def __init__(self):
		super(Logits, self).__init__()

	def forward(self, out_s, out_t):
		loss = F.mse_loss(out_s, out_t)

		return loss

class TTM(nn.Module):
    def __init__(self, l=0.1):
        super().__init__()
        self.l = l

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s, dim=1)
        p_t = torch.pow(torch.softmax(y_t, dim=1), self.l)
        norm = torch.sum(p_t, dim=1)
        p_t = p_t / norm.unsqueeze(1)
        KL = torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=1)
        loss = torch.mean(KL)

        return loss
    
class SP(nn.Module):
	'''
	Similarity-Preserving Knowledge Distillation
	https://arxiv.org/pdf/1907.09682.pdf
	'''
	def __init__(self):
		super(SP, self).__init__()

	def forward(self, fm_s, fm_t):
		fm_s = fm_s.view(fm_s.size(0), -1)
		G_s  = torch.mm(fm_s, fm_s.t())
		norm_G_s = F.normalize(G_s, p=2, dim=1)

		fm_t = fm_t.view(fm_t.size(0), -1)
		G_t  = torch.mm(fm_t, fm_t.t())
		norm_G_t = F.normalize(G_t, p=2, dim=1)

		loss = F.mse_loss(norm_G_s, norm_G_t)

		return loss
     
def lyapunov_kd(ts1, ts2, window_size=10, epsilon=1e-6):
    """
    Compute the Lyapunov coefficient to measure similarity between two multivariate time series.

    Parameters:
    - ts1, ts2: Two multivariate time series (tensors) of shape [batch_size, time_steps, features].
    - window_size: Length of the local window for Lyapunov computation.
    - epsilon: Small value to avoid division by zero.

    Returns:
    - lyapunov_coefficients: Tensor of average Lyapunov coefficients for each batch.
    """
    # Ensure the time series have the same shape
    assert ts1.shape == ts2.shape, "The two time series must have the same shape."
    batch_size, time_steps, features = ts1.shape

    # Check if the time series is long enough for the given window size
    if time_steps < window_size:
        raise ValueError("Time series length must be greater than the window size.")

    # Initialize storage for Lyapunov coefficients
    lyapunov_values = []

    # Loop over windows
    for i in range(time_steps - window_size):
        # Extract local windows for both time series
        segment_ts1 = ts1[:, i:i + window_size, :]  # Shape: [batch_size, window_size, features]
        segment_ts2 = ts2[:, i:i + window_size, :]  # Shape: [batch_size, window_size, features]

        # Compute initial and final distances within the window for each feature
        initial_distances = torch.norm(segment_ts1[:, :-1, :] - segment_ts2[:, :-1, :], dim=1) + epsilon  # Shape: [batch_size, window_size-1]
        final_distances = torch.norm(segment_ts1[:, 1:, :] - segment_ts2[:, 1:, :], dim=1) + epsilon       # Shape: [batch_size, window_size-1]
        #print(initial_distances.shape)
        # Compute log growth rates
        log_growth_rates = torch.log(final_distances / initial_distances)  # Shape: [batch_size, window_size-1]
        #print(log_growth_rates.shape)
        # Average log growth rates over the window
        avg_log_growth = torch.mean(log_growth_rates, dim=1)  # Shape: [batch_size]
        #print(avg_log_growth.shape)
        # Store the results
        lyapunov_values.append(avg_log_growth)

    # Stack results from all windows and compute the final average for each batch
    lyapunov_values = torch.stack(lyapunov_values, dim=1)  # Shape: [batch_size, num_windows]
    #print(lyapunov_values.shape)
    lyapunov_coefficients = torch.sum(torch.abs(lyapunov_values), dim=1)  # Shape: [batch_size]

    return lyapunov_coefficients

class SSLClassifier(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.ssl_encoder = UniSpeechSatModel.from_pretrained("microsoft/unispeech-sat-base-plus")
        self.ssl_encoder.requires_grad_(False)
        self.ssl_encoder.eval()
        self.HConv_Interface = SSL_Interface.interfaces.HierarchicalConvInterface(
            SSL_Interface.configs.HierarchicalConvInterfaceConfig(
                upstream_feat_dim=768,
                upstream_layer_num=7,
                normalize=False,
                conv_kernel_size=5,
                conv_kernel_stride=3,
                output_dim=768
            )
        )

        self.asp = MultiHeadAttentionPooling(768)
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512), nn.ReLU(), nn.Linear(512, 2)
        )
        self.config = config
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]))
        self.training_step_outputs = []
        self.validation_step_outputs = []
        _, self.d_meta = self._get_list_IDs(self.config['evaluation']['protocol_path'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.ssl_encoder(x, output_hidden_states=True)
        x = torch.stack(x.hidden_states[:7], dim=1)
        x = x.permute(1, 0, 2, -1)
        x = self.HConv_Interface(x)
        x = self.asp(x.permute(0, 2, 1))
        x = self.classifier[0](x.squeeze())
        x = self.classifier[1](x)
        x = self.classifier[-1](x)
        x = F.log_softmax(x, dim=-1)
        return x

    def on_epoch_end(self, outputs, phase):
        all_scores = []
        all_labels = []
        all_losses = []
        target_scores = []
        non_target_scores = []
        
        for preds, labels, loss in outputs:
            all_scores.append(preds)
            all_labels.append(labels)
            all_losses.append(loss)
            log_probs = F.log_softmax(preds, dim=-1)
            llr = log_probs[:, 1] - log_probs[:, 0]
            target_scores.extend(llr[labels == 1].tolist())
            non_target_scores.extend(llr[labels == 0].tolist())
        
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        all_losses = torch.stack(all_losses)
        
        eer, _ = compute_eer(target_scores, non_target_scores)
        all_scores = F.softmax(all_scores, dim=-1)
        self.accuracy(torch.argmax(all_scores, 1), all_labels)
        
        self.log_dict(
            {f"{phase}_eer": eer * 100,
             f"{phase}_loss": all_losses.mean(),
             f"{phase}_accuracy": self.accuracy.compute()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.training_step_outputs.append((scores, y, loss))

        return loss

    def on_train_epoch_end(self):
        self.on_epoch_end(self.training_step_outputs, phase="train")
        self.training_step_outputs.clear()
        self.accuracy.reset()


    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.validation_step_outputs.append((scores, y, loss))

        return loss
    
    def on_validation_epoch_end(self):
        self.on_epoch_end(self.validation_step_outputs, phase="val")
        self.validation_step_outputs.clear()
        self.accuracy.reset()

    def test_step(self, batch, batch_idx):
        self._produce_evaluation_file(batch, batch_idx)

    def _get_list_IDs(self, input_file):
        delimiter = ','
        list_IDs = []
        d_meta = {}
        with open(input_file, 'r') as infile:
            reader = csv.reader(infile, delimiter=delimiter)
            first_row = next(reader)
            if 'file_name' in first_row or 'label' in first_row:
                print("Skipping header row in the input file.")
            else:
                reader = [first_row] + list(reader)
            total_lines = sum(1 for _ in open(input_file)) - 1
            infile.seek(0)
            for row in tqdm(reader, total=total_lines, desc=f"Processing Ids"):
                try:
                    file_name, label = row
                    d_meta[file_name] = 1 if label == "bonafide" else 0
                    list_IDs.append(file_name)
                except ValueError as e:
                    print(f"Skipping malformed row: {row}. Error: {e}")

        return list_IDs, d_meta

    def _produce_evaluation_file(self, batch, batch_idx):
        x, utt_id = batch
        fname_list = []
        score_list = []
        out = self(x)
        out = F.log_softmax(out, dim=-1)
        ss = out[:, 0]
        bs = out[:, 1]
        llr = bs - ss
        if self.config['evaluation']['task'] == "asvspoof":
            utt_id = tuple(item.split('/')[-1].split('.')[0] for item in utt_id)
        fname_list.extend(utt_id)
        score_list.extend(llr.tolist())
            
        with open(self.config['evaluation']['output_score_file'], "a+") as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write("{} {}\n".format(f, cm))
        fh.close()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['training']['optimizer_gamma'])
        return [optimizer], [lr_scheduler]
