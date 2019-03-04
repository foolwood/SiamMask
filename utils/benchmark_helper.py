# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join, realpath, dirname, exists, isdir
from os import listdir
import logging
import glob
import numpy as np
import json
from collections import OrderedDict


def get_dataset_zoo():
    root = realpath(join(dirname(__file__), '../data'))
    zoos = listdir(root)

    def valid(x):
        y = join(root, x)
        if not isdir(y): return False

        return exists(join(y, 'list.txt')) \
               or exists(join(y, 'train', 'meta.json'))\
               or exists(join(y, 'ImageSets', '2016', 'val.txt'))

    zoos = list(filter(valid, zoos))
    return zoos


dataset_zoo = get_dataset_zoo()


def load_dataset(dataset):
    info = OrderedDict()
    if 'VOT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../data', dataset)
        if not exists(base_path):
            logging.error("Please download test dataset!!!")
            exit()
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            if gt.shape[1] == 4:
                gt = np.column_stack((gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3]-1,
                                      gt[:, 0] + gt[:, 2]-1, gt[:, 1] + gt[:, 3]-1, gt[:, 0] + gt[:, 2]-1, gt[:, 1]))
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'DAVIS' in dataset:
        base_path = join(realpath(dirname(__file__)), '../data', 'DAVIS')
        list_path = join(realpath(dirname(__file__)), '../data', 'DAVIS', 'ImageSets', dataset[-4:], 'val.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        for video in videos:
            info[video] = {}
            info[video]['anno_files'] = sorted(glob.glob(join(base_path, 'Annotations/480p', video, '*.png')))
            info[video]['image_files'] = sorted(glob.glob(join(base_path, 'JPEGImages/480p', video, '*.jpg')))
            info[video]['name'] = video
    elif 'ytb_vos' in dataset:
        base_path = join(realpath(dirname(__file__)), '../data', 'ytb_vos', 'valid')
        json_path = join(realpath(dirname(__file__)), '../data', 'ytb_vos', 'valid', 'meta.json')
        meta = json.load(open(json_path, 'r'))
        meta = meta['videos']
        info = dict()
        for v in meta.keys():
            objects = meta[v]['objects']
            frames = []
            anno_frames = []
            info[v] = dict()
            for obj in objects:
                frames += objects[obj]['frames']
                anno_frames += [objects[obj]['frames'][0]]
            frames = sorted(np.unique(frames))
            info[v]['anno_files'] = [join(base_path, 'Annotations', v, im_f+'.png') for im_f in frames]
            info[v]['anno_init_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in anno_frames]
            info[v]['image_files'] = [join(base_path, 'JPEGImages', v, im_f+'.jpg') for im_f in frames]
            info[v]['name'] = v

            info[v]['start_frame'] = dict()
            info[v]['end_frame'] = dict()
            for obj in objects:
                start_file = objects[obj]['frames'][0]
                end_file = objects[obj]['frames'][-1]
                info[v]['start_frame'][obj] = frames.index(start_file)
                info[v]['end_frame'][obj] = frames.index(end_file)
    else:
        logging.error('Not support')
        exit()
    return info
