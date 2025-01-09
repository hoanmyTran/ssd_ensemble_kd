# Ensemble Knowledge Distillation for Spoofed Speech Detection
For installation of SSL Interface, follow github repository: https://github.com/atosystem/SSL_Interface/tree/main

If there is any error, change the "view" to "reshape" in SSL_Interface repository


This repository uses pytorch lightning, install the rest if needed with pip install pytorch-lightning: https://pypi.org/project/pytorch-lightning/ .

```
pip install pytorch-lightning
```
## Training

For training the **teacher** or **student**, change the import file in **main.py**

To launch the training, modify the **config.yaml** file:
- Change *protocol_path* for both *train* and *val* in dataset
The protocole file path should be under the form without headers
*file_name,label*
which *file_name* is the path to the *file_name*.

Example:

```
../dataset/LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac,bonafide
../dataset/LA/ASVspoof2019_LA_train/flac/LA_T_1271820.flac,bonafide
```

```
python main.py
```
## Evaluation

To evaluate, modify the **config.yaml** file:
- Change the *eval: true* in evaluation
- Change the *protocol_path* to the protocole file path.
- Provide the checkpoint for *checkpoint* in *model*

The protocole file path should be under the form
*file_name,label*

Example:

```
file_name,label
../dataset/ASVspoof2021_DF_eval/flac/DF_E_2000049.flac,spoof
../dataset/ASVspoof2021_DF_eval/flac/DF_E_2000053.flac,bonafide
```


which *file_name* is the path to the *file_name*.

For ASVspoof evaluation, change *task* in **config.yaml** file to "asvspoof", for the rest, keep as *null*

Then run:
```
python main.py
```


Change the configuration in **config.yaml** file if necessary