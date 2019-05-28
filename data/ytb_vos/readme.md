# Preprocessing [Youtube-VOS](https://youtube-vos.org/dataset/download)

### Download raw images and annotations ([website](https://youtube-vos.org/dataset/download), 8.3G)

````shell
python download_from_gdrive.py https://drive.google.com/uc?id=18S_db1cFgSD1RsMsofJLkd6SyR9opk6a --output train.zip
unzip ./train.zip
python parse_ytb_vos.py  # really slow
````

### Crop & Generate data info (10 min)

````shell
#python par_crop.py [crop_size] [num_threads]
python par_crop.py 511 12
python gen_json.py
````
