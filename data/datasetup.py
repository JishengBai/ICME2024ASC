import os
import urllib.request

dataset_url_zenodo = ["", ""]

os.makedirs("data", exist_ok=True)

for url in dataset_url_zenodo:
    if not os.path.exists(url):
        print("Downloading dataset from url: {}".format(url))
        urllib.request.urlretrieve(
            url,
            "data",
        )

os.system("unzip data/*.zip -d data/")
os.system("python3 data/feature_extraction.py")
