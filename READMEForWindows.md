[Linux](https://github.com/zumrudu-anka/SiamMask/blob/master/README.md) | Windows

## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Testing](#testing)
4. [TODO LIST](#todo-list)
5. [References](#references)

## Environment Setup
- Clone the [Repository](https://github.com/foolwood/SiamMask)
```
git clone https://github.com/foolwood/SiamMask.git
set SiamMask=%cd%
```
- Setup Python Environment
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt 
python make.py
```
> If you see any error when install packages. Check it and please install up-to-date versions of the packages
- Add the project to your PYTHONPATH
```
set PYTHONPATH=%PYTHONPATH%;<your project path>
```
## Demo
- [Setup](#environment-setup) your environment
- Download the SiamMask model
```
cd <your project directory>/experiments/siammask_sharp
python downloadSiamMaskModel.py
```
- Change 26. line `from custom import Custom` to `from experiments.siammask_sharp.custom import Custom`
- Run demo.py
```
cd <your project directory>/experiments/siammask_sharp
set PYTHONPATH=%PYTHONPATH%;<your project path>
python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json
```
## Testing
- [Setup](#environment-setup) your environment
- Download test data
```
cd <your project directory>/data
python get_test_data.py
```
- Download pretrained models
```
cd <your project directory>/experiments/siammask_sharp
python downloadSiamMaskModel.py
python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json --base_path ../../data/VOT2019/drone_across
```

## TODO LIST

- Evaluate performance on [VOT](http://www.votchallenge.net/)
```
....
```

- Evaluate performance on [DAVIS](https://davischallenge.org/) (less than 50s)
```
....
```
- Evaluate performance on [Youtube-VOS](https://youtube-vos.org/) (need download data from [website](https://youtube-vos.org/dataset/download))
```
....
```

- ## Training

- ### Training Data 
  - Download the [Youtube-VOS](https://youtube-vos.org/dataset/download/), 
[COCO](http://cocodataset.org/#download), 
[ImageNet-DET](http://image-net.org/challenges/LSVRC/2015/), 
and [ImageNet-VID](http://image-net.org/challenges/LSVRC/2015/).
  - Preprocess each datasets according the [readme](data/coco/readme.md) files.

- ### Download the pre-trained model (174 MB)
  (This model was trained on the ImageNet-1k Dataset)
  ```
  cd $SiamMask/experiments
  wget http://www.robots.ox.ac.uk/~qwang/resnet.model
  ls | grep siam | xargs -I {} cp resnet.model {}
  ```

- ### Training SiamMask base model
  - [Setup](#environment-setup) your environment
  - From the experiment directory, run
  ```
  cd $SiamMask/experiments/siammask_base/
  bash run.sh
  ```
  - Training takes about 10 hours in our 4 Tesla V100 GPUs.
  - If you experience out-of-memory errors, you can reduce the batch size in `run.sh`.
  - You can view progress on Tensorboard (logs are at <experiment\_dir>/logs/)
  - After training, you can test checkpoints on VOT dataset.
  ```shell
  bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4  # test all snapshots with 4 GPUs
  ```
  - Select best model for hyperparametric search.
  ```shell
  #bash test_all.sh -m [best_test_model] -d VOT2018 -n [thread_num] -g [gpu_num] # 8 threads with 4 GPUS
  bash test_all.sh -m snapshot/checkpoint_e12.pth -d VOT2018 -n 8 -g 4 # 8 threads with 4 GPUS
  ```

- ### Training SiamMask model with the Refine module
  - [Setup](#environment-setup) your environment
  - In the experiment file, train with the best SiamMask base model
  ```
  cd $SiamMask/experiments/siammask_sharp
  bash run.sh <best_base_model>
  bash run.sh checkpoint_e12.pth
  ```
  - You can view progress on Tensorboard (logs are at <experiment\_dir>/logs/)
  - After training, you can test checkpoints on VOT dataset
  ```shell
  bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4
  ```

- ### Training SiamRPN++ model (*unofficial*)
  - [Setup](#environment-setup) your environment
  - From the experiment directory, run
  ```
  cd $SiamMask/experiments/siamrpn_resnet
  bash run.sh
  ```
  - You can view progress on Tensorboard (logs are at <experiment\_dir>/logs/)
  - After training, you can test checkpoints on VOT dataset
  ```shell
  bash test_all.sh -h
  bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4
  ```

### References
[SiamMask](https://github.com/foolwood/SiamMask)
