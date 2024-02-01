# Baseline for IEEE ICME 2024 Grand Challenge: Semi-supervised Acoustic Scene Classification under Domain Shift

## Challenge website
[ICME2024 GC](https://2024.ieeeicme.org/grand-challenge-proposals/)  
[Challenge website](https://ascchallenge.xshengyun.com/)

## Official Baseline

### Step 1: Python Running Environment
```shell
conda create -n ASC python=3.10
conda activate ASC
git clone git@github.com:JishengBai/ICME2024ASC.git; cd ICME2024ASC
pip install -r requirement.txt
```  

### Step 2: Setup Dataset
```shell
python3 data/datasetup.py
```

### Step3: Train and evaluate model

```shell
# Model training
python train.py

# Model testing
python test.py
```

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




