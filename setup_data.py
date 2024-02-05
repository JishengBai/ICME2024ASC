import os
import urllib.request

os.makedirs("data", exist_ok=True)

if not os.path.exists("data/ICME2024_GC_ASC_dev.zip") or os.path.getsize("data/ICME2024_GC_ASC_dev.zip") < 5271935461: # The dev dataset is 5271935462
    url = "https://zenodo.org/records/10616533/files/ICME2024_GC_ASC_dev.zip"
    print("Downloading dev dataset from url: {}".format(url))
    urllib.request.urlretrieve(
        url,
        "data/ICME2024_GC_ASC_dev.zip",
    )

# if not os.path.exists("data/ICME2024_GC_ASC_eval.zip") or os.path.getsize("data/ICME2024_GC_ASC_eval.zip") < 694982408: # The eval dataset is 694982409
#     url = ""
#     print("Downloading eval dataset from url: {}".format(url))
#     urllib.request.urlretrieve(
#         url,
#         "data",
#     )

os.system("unzip -n -q data/ICME2024_GC_ASC_dev.zip -d data/")
# os.system("unzip -n -q data/ICME2024_GC_ASC_eval.zip -d data/")
os.system("python3 feature_extraction.py")
