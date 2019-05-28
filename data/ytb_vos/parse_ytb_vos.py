# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import h5py
import json
import os
import scipy.misc
import sys
import numpy as np
import cv2
from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--outdir', default='./', type=str,
                        help="output dir for json files")
    parser.add_argument('--datadir', default='./', type=str,
                        help="data dir for annotations to be converted")
    return parser.parse_args()


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]
    return boxes_from_polys


class Instance(object):
    instID     = 0
    pixelCount = 0

    def __init__(self, imgNp, instID):
        if (instID ==0 ):
            return
        self.instID     = int(instID)
        self.pixelCount = int(self.getInstancePixels(imgNp, instID))

    def getInstancePixels(self, imgNp, instLabel):
        return (imgNp == instLabel).sum()

    def toDict(self):
        buildDict = {}
        buildDict["instID"]     = self.instID
        buildDict["pixelCount"] = self.pixelCount
        return buildDict

    def __str__(self):
        return "("+str(self.instID)+")"


def convert_ytb_vos(data_dir, out_dir):
    sets = ['train']
    ann_dirs = ['train/Annotations/']
    json_name = 'instances_%s.json'
    num_obj = 0
    num_ann = 0
    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        ann_dir = os.path.join(data_dir, ann_dir)
        json_ann = json.load(open(os.path.join(ann_dir, '../meta.json')))
        for vid, video in enumerate(json_ann['videos']):
            v = json_ann['videos'][video]
            frames = []
            for obj in v['objects']:
                o = v['objects'][obj]
                frames.extend(o['frames'])
            frames = sorted(set(frames))

            annotations = []
            instanceIds = []
            for frame in frames:
                file_name = join(video, frame)
                fullname = os.path.join(ann_dir, file_name+'.png')
                img = cv2.imread(fullname, 0)
                h, w = img.shape[:2]

                objects = dict()
                for instanceId in np.unique(img):
                    if instanceId == 0:
                        continue
                    instanceObj = Instance(img, instanceId)
                    instanceObj_dict = instanceObj.toDict()
                    mask = (img == instanceId).astype(np.uint8)
                    _, contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    polygons = [c.reshape(-1).tolist() for c in contour]
                    instanceObj_dict['contours'] = [p for p in polygons if len(p) > 4]
                    if len(instanceObj_dict['contours']) and instanceObj_dict['pixelCount'] > 1000:
                        objects[instanceId] = instanceObj_dict
                    # else:
                    #     cv2.imshow("disappear?", mask)
                    #     cv2.waitKey(0)

                for objId in objects:
                    if len(objects[objId]) == 0:
                        continue
                    obj = objects[objId]
                    len_p = [len(p) for p in obj['contours']]
                    if min(len_p) <= 4:
                        print('Warning: invalid contours.')
                        continue  # skip non-instance categories

                    ann = dict()
                    ann['h'] = h
                    ann['w'] = w
                    ann['file_name'] = file_name
                    ann['id'] = int(objId)
                    # ann['segmentation'] = obj['contours']
                    # ann['iscrowd'] = 0
                    ann['area'] = obj['pixelCount']
                    ann['bbox'] = xyxy_to_xywh(polys_to_boxes([obj['contours']])).tolist()[0]

                    annotations.append(ann)
                    instanceIds.append(objId)
                    num_ann += 1
            instanceIds = sorted(set(instanceIds))
            num_obj += len(instanceIds)
            video_ann = {str(iId): [] for iId in instanceIds}
            for ann in annotations:
                video_ann[str(ann['id'])].append(ann)

            ann_dict[video] = video_ann
            if vid % 50 == 0 and vid != 0:
                print("process: %d video" % (vid+1))

        print("Num Videos: %d" % len(ann_dict))
        print("Num Objects: %d" % num_obj)
        print("Num Annotations: %d" % num_ann)

        items = list(ann_dict.items())
        train_dict = dict(items[:3000])
        val_dict = dict(items[3000:])
        with open(os.path.join(out_dir, json_name % 'train'), 'w') as outfile:
            json.dump(train_dict, outfile)

        with open(os.path.join(out_dir, json_name % 'val'), 'w') as outfile:
            json.dump(val_dict, outfile)


if __name__ == '__main__':
    args = parse_args()
    convert_ytb_vos(args.datadir, args.outdir)
