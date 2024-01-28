step1: conda create -n ASC python=3.7
conda activate ASC
pip install -r requirement.txt

step2: Download ICME2024 ASC GC development dataset

step3: set paths and paramters in config.py

step4: python feature_extraction.py

step5: python train.py

step6: Download ICME2024 ASC GC evaluation dataset

step7: set paths and paramters in config.py

step8: python feature_extraction.py

step9: python test.py








