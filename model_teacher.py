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

class SSLClassifier(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.ssl_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.ssl_encoder.requires_grad_(False)
        self.ssl_encoder.eval()
        self.HConv_Interface = SSL_Interface.interfaces.HierarchicalConvInterface(
            SSL_Interface.configs.HierarchicalConvInterfaceConfig(
                upstream_feat_dim=1024,
                upstream_layer_num=13,
                normalize=False,
                conv_kernel_size=5,
                conv_kernel_stride=3,
                output_dim=1024
            )
        )

        self.asp = MultiHeadAttentionPooling(1024)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 2)
        )
        self.config = config
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]))
        self.training_step_outputs = []
        self.validation_step_outputs = []
        _, self.d_meta = self._get_list_IDs(self.config['evaluation']['protocol_path'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.feature_extractor(x, output_hidden_states=True)
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
