import os
import urllib.request

os.makedirs("data", exist_ok=True)

if not os.path.exists("data/ICME2024_GC_ASC_dev.zip"):
    url = ""
    print("Downloading dev dataset from url: {}".format(url))
    urllib.request.urlretrieve(
        url,
        "data",
    )

if not os.path.exists("data/ICME2024_GC_ASC_eval.zip"):
    url = ""
    print("Downloading eval dataset from url: {}".format(url))
    urllib.request.urlretrieve(
        url,
        "data",
    )

if not os.path.exists("data/ICME2024_GC_ASC_dev") or not os.path.exists("data/ICME2024_GC_ASC_eval"):
    os.system("unzip data/*.zip -n -d data/")

os.system("python3 feature_extraction.py")
