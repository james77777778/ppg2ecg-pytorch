# PPG2ECG
The official implementation of the paper "Reconstructing QRS Complex from PPG by Transformed Attentional Neural Networks"

## Results
Graph Abstract
![](doc/imgs/chiu1.png)

Model Architecture
![](doc/imgs/chiu2.png)

Reconstruction Visualization
![](doc/imgs/chiu4.png)

## Dataset
https://drive.google.com/file/d/15dxbpi4FH7lJbRFZwyREyX4V0VKDxXNs/view?usp=sharing
```bash
mkdir data
unzip dataset.zip -d data
```
and you should have following structure
```bash
data/
├── bidmc
│   ├── bidmc_csv
│   ├── bidmc-filtered
│   ├── bidmc-filtered-test
│   └── bidmc-filtered-train
└── uqvitalsigns
    ├── uqvitalsignsdata
    ├── uqvitalsignsdata-test
    └── uqvitalsignsdata-train
```

## Environment
- Ubuntu 18.04
- python 3.6
- pytorch 1.2
...
You can check it yourself in requirements.txt

## Installation
```bash
# in your environment with pip
pip install -r requirements.txt
```

## Usage
```bash
# run UQVIT dataset with full model
python3 train.py --flagfile config/UQVIT.cfg

# run UQVIT dataset with LSTM baseline model
python3 train.py --flagfile config/UQVIT_LSTM.cfg

# run BIDMC dataset with full model
python3 train.py --flagfile config/BIDMC.cfg
```
all the training parameters are included in config files.

## Tensorboard
```bash
tensorboard --logdir logs
```
