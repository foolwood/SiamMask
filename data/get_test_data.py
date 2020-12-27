import os
from shutil import copyfile
import requests

abspath = os.path.abspath(__file__)
projectDir = os.path.dirname(abspath)
trackDat = projectDir + "/trackdat"

if not os.path.exists(trackDat):
    os.system("git clone https://github.com/jvlmdr/trackdat.git")
else:
    print("Loading...")

scripts = trackDat + "/scripts"
os.chdir(scripts)
os.system("download_vot.py 2016 dl/vot2016")
os.system("download_vot.py 2018 dl/vot2018")
os.system("download_vot.py 2019 dl/vot2019")
os.system("unpack_vot.py 2016 dl/vot2016_main VOT2016")
os.system("unpack_vot.py 2018 dl/vot2018_main VOT2018")
os.system("unpack_vot.py 2019 dl/vot2019_main VOT2019")

Vot2016ListTxtPath = trackDat + "/dl/vot2016_main/list.txt"
Vot2018ListTxtPath = trackDat + "/dl/vot2018_main/list.txt"
Vot2019ListTxtPath = trackDat + "/dl/vot2019_main/list.txt"

copyfile(Vot2016ListTxtPath, projectDir + "/VOT2016/list.txt")
copyfile(Vot2018ListTxtPath, projectDir + "/VOT2018/list.txt")
copyfile(Vot2019ListTxtPath, projectDir + "/VOT2019/list.txt")

os.chdir(projectDir)

base_url = "http://www.robots.ox.ac.uk/~qwang/"
urlArray = [
    "VOT2016.json",
    "VOT2018.json",
]

for url in urlArray:
    response = requests.get(base_url + url, allow_redirects = True)
    with open(url, "wb") as jsonFile:
        jsonFile.write(response.content)




# ! TODOOOOO

# # json file for eval toolkit
# wget http://www.robots.ox.ac.uk/~qwang/VOT2016.json
# wget http://www.robots.ox.ac.uk/~qwang/VOT2018.json
# python create_json.py VOT2019

# # DAVIS
# wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
# unzip DAVIS-2017-trainval-480p.zip
# ln -s ./DAVIS ./DAVIS2016
# ln -s ./DAVIS ./DAVIS2017


# # Youtube-VOS
