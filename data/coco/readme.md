# Preprocessing COCO

### Download raw images and annotations

````shell
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip ./train2017.zip
unzip ./val2017.zip
unzip ./annotations_trainval2017.zip
cd pycocotools && make && cd ..
````

### Crop & Generate data info (~20 min)

````shell
#python par_crop.py -h
python par_crop.py --enable_mask --num_threads 24
python gen_json.py
````
