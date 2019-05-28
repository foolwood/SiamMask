# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from pycocotools.coco import COCO
from os.path import join
import json


dataDir = '.'
for data_subset in ['val2017', 'train2017']:
    dataset = dict()
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, data_subset)
    coco = COCO(annFile)
    n_imgs = len(coco.imgs)
    for n, img_id in enumerate(coco.imgs):
        print('subset: {} image id: {:04d} / {:04d}'.format(data_subset, n, n_imgs))
        img = coco.loadImgs(img_id)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        crop_base_path = join(data_subset, img['file_name'].split('/')[-1].split('.')[0])
        
        if len(anns) > 0:
            dataset[crop_base_path] = dict()

        for track_id, ann in enumerate(anns):
            rect = ann['bbox']
            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                continue
            bbox = [rect[0], rect[1], rect[0]+rect[2]-1, rect[1]+rect[3]-1]  # x1,y1,x2,y2

            dataset[crop_base_path]['{:02d}'.format(track_id)] = {'000000': bbox}

    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open('{}.json'.format(data_subset), 'w'), indent=4, sort_keys=True)
    print('done!')

