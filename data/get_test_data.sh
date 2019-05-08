#!/bin/bash
# VOT
git clone https://github.com/jvlmdr/trackdat.git
cd trackdat
VOT_YEAR=2016 bash scripts/download_vot.sh dl/vot2016
VOT_YEAR=2018 bash scripts/download_vot.sh dl/vot2018
VOT_YEAR=2019 bash scripts/download_vot.sh dl/vot2019
bash scripts/unpack_vot.sh dl/vot2016 ../VOT2016
bash scripts/unpack_vot.sh dl/vot2018 ../VOT2018
bash scripts/unpack_vot.sh dl/vot2019 ../VOT2019
cp dl/vot2016/list.txt ../VOT2016/
cp dl/vot2018/list.txt ../VOT2018/
cp dl/vot2019/list.txt ../VOT2019/
cd .. && rm -rf ./trackdat

# json file for eval toolkit
wget http://www.robots.ox.ac.uk/~qwang/VOT2016.json
wget http://www.robots.ox.ac.uk/~qwang/VOT2018.json
python create_json.py VOT2019

# DAVIS
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
ln -s ./DAVIS ./DAVIS2016
ln -s ./DAVIS ./DAVIS2017


# Youtube-VOS
