# Preprocessing DET(Object detection)
Large Scale Visual Recognition Challenge 2015 (ILSVRC2015)

### Download dataset (49GB)

````shell
wget http://image-net.org/image/ILSVRC2015/ILSVRC2015_DET.tar.gz
tar -xzvf ./ILSVRC2015_DET.tar.gz
````

### Crop & Generate data info (10 min)

````shell
#python par_crop.py [crop_size] [num_threads]
python par_crop.py 511 12
python gen_json.py
````
