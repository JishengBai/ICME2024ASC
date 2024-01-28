# Baseline for IEEE ICME 2024 Grand Challenge
# Semi-supervised Acoustic Scene Classification under Domain Shift

## Dataset
The development and evaluation datasets will be released in [zenodo](https://zenodo.org/records/10558800) when the challenge starts.

## Challenge website
[ICME2024 GC](https://2024.ieeeicme.org/grand-challenge-proposals/)  
[Challenge website](https://ascchallenge.xshengyun.com/)

## Run the code
Step 1:  
```
conda create -n ASC python=3.7
conda activate ASC
pip install -r requirement.txt
```  
Step 2: Download ICME2024 ASC GC development dataset  
Step 3: set paths and parameters in `config.py`  
Step 4: `python feature_extraction.py`  
Step 5: `python train.py`  
Step 6: Download ICME2024 ASC GC evaluation dataset  
Step 7: set paths and parameters in config.py  
Step 8: `python feature_extraction.py`  
Step 9: `python test.py`

## Cite






