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

# Unzip zip files in the ./data folder, with os.system
os.system("unzip data/*.zip -d data/")
os.system("python3 feature_extraction.py")
