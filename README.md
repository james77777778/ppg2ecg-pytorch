# PPG2ECG
The official implementation of the paper "Reconstructing QRS Complex from PPG by Transformed Attentional Neural Networks"
[https://ieeexplore.ieee.org/document/9109576](https://ieeexplore.ieee.org/document/9109576)

## Results
Graph Abstract
![](doc/imgs/chiu1.png)

Model Architecture
![](doc/imgs/chiu2.png)

<!-- Reconstruction Visualization
![](doc/imgs/chiu4.png) -->

## Dataset
Download the dataset:

- https://drive.google.com/file/d/1UwuHRKkC0YPbDAFIYvFJlFmU6_3zgjcJ/view
- https://github.com/james77777778/ppg2ecg-pytorch/releases/download/dataset/dataset.zip

And follow the instruction:
```bash
mkdir data
unzip dataset.zip -d data
```

After that, you should have following data structure:
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

The main dataset we used in this paper can be found at

[The University of Queensland Vital Signs Dataset](https://outbox.eait.uq.edu.au/uqdliu3/uqvitalsignsdataset/index.html)

## Pretrained Model (UQVIT)
Download the model weights and usually we put it in `./weights`.

~~https://drive.google.com/file/d/10aYWNkgaGCz1zU6--kN3yaW6L_9BzkhQ/view?usp=sharing~~

(Sorry for the inconvience. The model weights are lost.)

## Environment
You can check it yourself in requirements.txt
- Ubuntu 18.04
- python 3.6
- pytorch 1.2
...

## Installation
```bash
# in your environment with pip
pip install -r requirements.txt
```

## Usage
All the training parameters are included in config files.
```bash
# run UQVIT dataset with full model
python3 train.py --flagfile config/UQVIT.cfg

# run UQVIT dataset with LSTM baseline model
python3 train.py --flagfile config/UQVIT_LSTM.cfg

# run BIDMC dataset with full model
python3 train.py --flagfile config/BIDMC.cfg
```

## Test for your own PPG data
Please see [EXAMPLE.md](doc/imgs/EXAMPLE.md).

Simple result:

![](doc/imgs/example.png)

## Tensorboard
```bash
tensorboard --logdir logs
```

## Citation
If you use this code for your research, please cite our papers.
```
@ARTICLE{ppg2ecg,
  author={H. -Y. {Chiu} and H. -H. {Shuai} and P. C. . -P. {Chao}},
  journal={IEEE Sensors Journal}, 
  title={Reconstructing QRS Complex From PPG by Transformed Attentional Neural Networks}, 
  year={2020},
  volume={20},
  number={20},
  pages={12374-12383},}
```
