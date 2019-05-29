# SiamMask

**NEW:** now including code for both training and inference!


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fast-online-object-tracking-and-segmentation/visual-object-tracking-vot201718)](https://paperswithcode.com/sota/visual-object-tracking-vot201718?p=fast-online-object-tracking-and-segmentation)

This is the official implementation with *training* code for SiamMask (CVPR2019). For technical details, please refer to:

**Fast Online Object Tracking and Segmentation: A Unifying Approach** <br />
[Qiang Wang](http://www.robots.ox.ac.uk/~qwang/)\*, [Li Zhang](http://www.robots.ox.ac.uk/~lz)\*, [Luca Bertinetto](http://www.robots.ox.ac.uk/~luca)\*, [Weiming Hu](https://scholar.google.com/citations?user=Wl4tl4QAAAAJ&hl=en), [Philip H.S. Torr](https://scholar.google.it/citations?user=kPxa2w0AAAAJ&hl=en&oi=ao) (\* denotes equal contribution) <br />
**CVPR 2019** <br />
**[[Paper](https://arxiv.org/abs/1812.05050)] [[Video](https://youtu.be/I_iOVrcpEBw)] [[Project Page](http://www.robots.ox.ac.uk/~qwang/SiamMask)]** <br />


<div align="center">
  <img src="http://www.robots.ox.ac.uk/~qwang/SiamMask/img/SiamMask.jpg" width="600px" />
</div>

### Bibtex
If you find this code useful, please consider citing:

```
@inproceedings{wang2019fast,
    title={Fast online object tracking and segmentation: A unifying approach},
    author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    year={2019}
}
```


## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Testing Models](#testing-models)
4. [Training Models](#training-models)

## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, CUDA 9.2, RTX 2080 GPUs

- Clone the repository 
```
git clone https://github.com/foolwood/SiamMask.git && cd SiamMask
export SiamMask=$PWD
```
- Setup python environment
```
conda create -n siammask python=3.6
source activate siammask
pip install -r requirements.txt
bash make.sh
```
- Add the project to your PYTHONPATH
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Demo
- [Setup](#environment-setup) your environment
- Download the SiamMask model
```shell
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Run `demo.py`

```shell
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json
```

<div align="center">
  <img src="http://www.robots.ox.ac.uk/~qwang/SiamMask/img/SiamMask_demo.gif" width="500px" />
</div>


## Testing
- [Setup](#environment-setup) your environment
- Download test data
```shell
cd $SiamMask/data
sudo apt-get install jq
bash get_test_data.sh
```
- Download pretrained models
```shell
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT_LD.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Evaluate performance on [VOT](http://www.votchallenge.net/)
```shell
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2016 0
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2018 0
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2019 0
bash test_mask_refine.sh config_vot18.json SiamMask_VOT_LD.pth VOT2016 0
bash test_mask_refine.sh config_vot18.json SiamMask_VOT_LD.pth VOT2018 0
python ../../tools/eval.py --dataset VOT2016 --tracker_prefix C --result_dir ./test/VOT2016
python ../../tools/eval.py --dataset VOT2018 --tracker_prefix C --result_dir ./test/VOT2018
python ../../tools/eval.py --dataset VOT2019 --tracker_prefix C --result_dir ./test/VOT2019
```
- Evaluate performance on [DAVIS](https://davischallenge.org/) (less than 50s)
```shell
bash test_mask_refine.sh config_davis.json SiamMask_DAVIS.pth DAVIS2016 0
bash test_mask_refine.sh config_davis.json SiamMask_DAVIS.pth DAVIS2017 0
```
- Evaluate performance on [Youtube-VOS](https://youtube-vos.org/) (need download data from [website](https://youtube-vos.org/dataset/download))
```shell
bash test_mask_refine.sh config_davis.json SiamMask_DAVIS.pth ytb_vos 0
```

### Results
These are the reproduction results from this repository. All results can be downloaded from our [project page](http://www.robots.ox.ac.uk/~qwang/SiamMask/).

|                           <sub>Tracker</sub>                           |      <sub>VOT2016</br>EAO /  A / R</sub>     |      <sub>VOT2018</br>EAO / A / R</sub>      |  <sub>DAVIS2016</br>J / F</sub>  |  <sub>DAVIS2017</br>J / F</sub>  |     <sub>Youtube-VOS</br>J_s / J_u / F_s / F_u</sub>     |     <sub>Speed</sub>     |
|:----------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|:--------------------------------:|:--------------------------------:|:--------------------------------------------------------:|:------------------------:|
| <sub>[SiamMask-box](http://www.robots.ox.ac.uk/~qwang/SiamMask/)</sub> |       <sub>0.412/0.623/0.233</sub>       |       <sub>0.363/0.584/0.300</sub>       |               - / -              |               - / -              |                      - / - / - / -                       | <sub>**77** FPS</sub> |
| <sub>[SiamMask](http://www.robots.ox.ac.uk/~qwang/SiamMask/)</sub> | <sub>**0.433**/**0.639**/**0.214**</sub> | <sub>**0.380**/**0.609**/**0.276**</sub> | <sub>**0.713**/**0.674**</sub> | <sub>**0.543**/**0.585**</sub> | <sub>**0.602**/**0.451**/**0.582**/**0.477**</sub> |   <sub>56 FPS</sub>   |
| <sub>[SiamMask-LD](http://www.robots.ox.ac.uk/~qwang/SiamMask/)</sub> | <sub>**0.455**/**0.634**/**0.219**</sub> | <sub>**0.423**/**0.615**/**0.248**</sub> | - / - | - / - | - / - / - / - | <sub>56 FPS</sub> |

**Note:** 
- Speed are tested on a NVIDIA RTX 2080. 
- `-box` reports an axis-aligned bounding box from the box branch.
- `-LD` means training with large dataset (ytb-bb+ytb-vos+vid+coco+det).


## Training

### Training Data 
- Download the [Youtube-VOS](https://youtube-vos.org/dataset/download/), 
[COCO](http://cocodataset.org/#download), 
[ImageNet-DET](http://image-net.org/challenges/LSVRC/2015/), 
and [ImageNet-VID](http://image-net.org/challenges/LSVRC/2015/).
- Preprocess each datasets according the [readme](data/coco/readme.md) files.

### Download the pre-trained model (174 MB)
(This model was trained on the ImageNet-1k Dataset)
```
cd $SiamMask/experiments
wget http://www.robots.ox.ac.uk/~qwang/resnet.model
ls | grep siam | xargs -I {} cp resnet.model {}
```

### Training SiamMask base model
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

### Training SiamMask model with the Refine module
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

### Training SiamRPN++ model (*unofficial*)
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

## License
Licensed under an MIT license.

