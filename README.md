# Semi-supervised Acoustic Scene Classification under Domain Shift

Baseline for IEEE ICME 2024 Grand Challenge.

This Challenge aims to push the boundaries of computational audition by tackling one of its most compelling problems: effectively classifying acoustic scenes under significant domain shifts.

## Challenge Website
[ICME2024 GC](https://2024.ieeeicme.org/grand-challenge-proposals/)  
[Challenge website](https://ascchallenge.xshengyun.com/)

## Official Baseline

**The baseline and dataset is under development and not ready yet.**

**The organization team will finalize the dataset and code around 5th Feb 2024, please stay tuned for the update.**

![main](pics/main.jpg)

### Step 1: Python Running Environment
```shell
conda create -n ASC python=3.10
conda activate ASC
git clone git@github.com:JishengBai/ICME2024ASC.git; cd ICME2024ASC
pip install -r requirement.txt
```  

### Step 2: Setup Dataset
This step includes dataset download, unzip, and feature extraction. 
```shell
# Takes about an hour
python3 setup_data.py
# Our dataset is available on Zenodo: xxx.
```

### Step3: Train and Evaluate Model

```shell
# Model training, which includes the following three steps:
# (1) Training with limited labels; (2) Pseudo labeling; (3) Model training with pseudo labels.
# In total the training process takes about 30 minutes on a single NVIDIA 2080 Ti.
python train.py

# Model testing.
python test.py
```
You can find an example training log [here](https://github.com/JishengBai/ICME2024ASC/blob/main/data/example_train.log)

## Cite
```bibtex
coming soon
```

## Organization
- Northwestern Polytechnical University, China
- Xi'an Lianfeng Acoustic Technologies Co., Ltd., China
- Nanyang Technological University, Singapore
- Institute of Acoustics, Chinese Academy of Sciences, China
- University of Surrey, UK




