import json
import os
import re
import numpy as np
import cv2

from glob import glob
from fire import Fire

def process(dataset_name):
    with open(os.path.join(dataset_name, 'list.txt'), 'r') as f:
        lines = f.readlines()
    videos = [x.strip() for x in lines]

    # if dataset_name == 'VOT2016':
    meta_data = {}
    tags = []
    for video in videos:
        with open(os.path.join(dataset_name, video, "groundtruth.txt"),'r') as f:
            gt_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]

        img_names = sorted(glob(os.path.join(dataset_name, video, 'color', '*.jpg')))
        if len(img_names) == 0:
            img_names = sorted(glob(os.path.join(dataset_name, video, '*.jpg')))
        im = cv2.imread(img_names[0])
        img_names = [x.split('/', 1)[1] for x in img_names]
        # tag
        if dataset_name in ['VOT2018', 'VOT2019']:
            tag_file = os.path.join(dataset_name, video, 'camera_motion.tag')
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    camera_motion = [int(x.strip()) for x in f.readlines()]
                    camera_motion += [0] * (len(gt_traj) - len(camera_motion))
            else:
                print("File not exists: ", tag_file)
                camera_motion = [] # [0] * len(gt_traj)

            tag_file = os.path.join(dataset_name, video, 'illum_change.tag')
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    illum_change = [int(x.strip()) for x in f.readlines()]
                    illum_change += [0] * (len(gt_traj) - len(illum_change))
            else:
                print("File not exists: ", tag_file)
                illum_change = [] # [0] * len(gt_traj)

            tag_file = os.path.join(dataset_name, video, 'motion_change.tag')
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    motion_change = [int(x.strip()) for x in f.readlines()]
                    motion_change  += [0] * (len(gt_traj) - len(motion_change))
            else:
                print("File not exists: ", tag_file)
                motion_change = [] # [0] * len(gt_traj)

            tag_file = os.path.join(dataset_name, video, 'size_change.tag')
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    size_change = [int(x.strip()) for x in f.readlines()]
                    size_change  += [0] * (len(gt_traj) - len(size_change))
            else:
                print("File not exists: ", tag_file)
                size_change  = [] # [0] * len(gt_traj)

            tag_file = os.path.join(dataset_name, video, 'occlusion.tag')
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    occlusion = [int(x.strip()) for x in f.readlines()]
                    occlusion  += [0] * (len(gt_traj) - len(occlusion))
            else:
                print("File not exists: ", tag_file)
                occlusion  = [] # [0] * len(gt_traj)
            img_files = os.path.join('VOT2019', )
            meta_data[video] = {'video_dir': video,
                                'init_rect': gt_traj[0],
                                'img_names': img_names,
                                'width': im.shape[1],
                                'height': im.shape[0],
                                'gt_rect': gt_traj,
                                'camera_motion': camera_motion,
                                'illum_change': illum_change,
                                'motion_change': motion_change,
                                'size_change': size_change,
                                'occlusion': occlusion}
        elif 'VOT2016' == dataset_name:
            tag_file = os.path.join(dataset_name, video, 'camera_motion.label')
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    camera_motion = [int(x.strip()) for x in f.readlines()]
                    camera_motion += [0] * (len(gt_traj) - len(camera_motion))
            else:
                print("File not exists: ", tag_file)
                camera_motion = [] # [0] * len(gt_traj)

            tag_file = os.path.join(dataset_name, video, 'illum_change.label')
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    illum_change = [int(x.strip()) for x in f.readlines()]
                    illum_change += [0] * (len(gt_traj) - len(illum_change))
            else:
                print("File not exists: ", tag_file)
                illum_change = [] # [0] * len(gt_traj)

            tag_file = os.path.join(dataset_name, video, 'motion_change.label')
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    motion_change = [int(x.strip()) for x in f.readlines()]
                    motion_change  += [0] * (len(gt_traj) - len(motion_change))
            else:
                print("File not exists: ", tag_file)
                motion_change = [] # [0] * len(gt_traj)

            tag_file = os.path.join(dataset_name, video, 'size_change.label')
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    size_change = [int(x.strip()) for x in f.readlines()]
                    size_change  += [0] * (len(gt_traj) - len(size_change))
            else:
                print("File not exists: ", tag_file)
                size_change  = [] # [0] * len(gt_traj)

            tag_file = os.path.join(dataset_name, video, 'occlusion.label')
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    occlusion = [int(x.strip()) for x in f.readlines()]
                    occlusion  += [0] * (len(gt_traj) - len(occlusion))
            else:
                print("File not exists: ", tag_file)
                occlusion  = [] # [0] * len(gt_traj)

            meta_data[video] = {'video_dir': video,
                                'init_rect': gt_traj[0],
                                'img_names': img_names,
                                'gt_rect': gt_traj,
                                'width': im.shape[1],
                                'height': im.shape[0],
                                'camera_motion': camera_motion,
                                'illum_change': illum_change,
                                'motion_change': motion_change,
                                'size_change': size_change,
                                'occlusion': occlusion}
        else:
            meta_data[video] = {'video_dir': video,
                                'init_rect': gt_traj[0],
                                'img_names': img_names,
                                'gt_rect': gt_traj,
                                'width': im.shape[1],
                                'height': im.shape[0]}


    json.dump(meta_data, open(dataset_name+'.json', 'w'))

if __name__ == '__main__':
    Fire(process)

