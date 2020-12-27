import sys
import os
import requests
import json
import zipfile
import io
import urllib.request, shutil

VOT_YEAR = sys.argv[1]
VOT_CHALLENGE = "main"
name = "vot" + VOT_YEAR + "_" + VOT_CHALLENGE

dl = f"../dl/{name}"

try:
    os.makedirs(dl)
except FileExistsError:
    print("Loading...")

abspath = os.path.abspath(__file__)
projectDir = os.path.dirname(abspath)

os.chdir(dl)

base_url = f"https://data.votchallenge.net/vot{VOT_YEAR}/{VOT_CHALLENGE}"

if os.path.exists("description.json"):
    pass
else:
    response = requests.get(base_url + "/description.json", allow_redirects = True)
    with open("description.json", "wb") as descriptionFile:
        descriptionFile.write(response.content)

with open("annotations.txt", "w") as annotationsFileOutput, open("color.txt", "w") as colorFileOutput, open("list.txt", "w") as listFileOutput:
    with open("description.json", "r") as description:
        data = json.load(description)
        for sequence in data["sequences"]:
            annotationsFileOutput.write(sequence["annotations"]["url"] + "\n")
            colorFileOutput.write(sequence["channels"]["color"]["url"] + "\n")
            listFileOutput.write(sequence["name"] + "\n")

try:
    os.makedirs("annotations")
except FileExistsError:
    print("Loading...")

try:
    os.makedirs("color")
except FileExistsError:
    print("Loading...")

os.chdir("./annotations")
with open("../annotations.txt", "r") as annotationsTxtFile:
    lines = annotationsTxtFile.readlines()
    for line in lines:
        url = f"{base_url}/{line}"
        response = requests.get(url, stream = True)
        with urllib.request.urlopen(url) as response, open(line.strip(), "wb") as downloadedAnnotation:
            shutil.copyfileobj(response, downloadedAnnotation)
            print("Test datas are downloading...")

os.chdir("../color")
with open("../color.txt", "r") as colorTxtFile:
    lines = colorTxtFile.readlines()
    for line in lines:
        url = f"{base_url}/{line}"
        response = requests.get(url, stream = True)
        with urllib.request.urlopen(url) as response, open(line[16:].strip(), "wb") as downloadedColor:
            shutil.copyfileobj(response, downloadedColor)
            print("Test datas are downloading...")
        
        # response = requests.get(url, allow_redirects = True)
        # with open(line[16:].strip(), "wb") as downloadedColor:
        #     downloadedColor.write(response.content)
        #     print("Test datas are downloading...")