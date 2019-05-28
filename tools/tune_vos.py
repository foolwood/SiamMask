# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import isfile, isdir, join

from utils.log_helper import init_log, add_file_handler
from utils.bbox_helper import cxy_wh_2_rect
from utils.load_helper import load_pretrain
from utils.benchmark_helper import load_dataset
from utils.average_meter_helper import IouMeter

import models as models
from tools.test import siamese_init, siamese_track
from utils.config_helper import load_config

thrs = np.arange(0.3, 0.81, 0.05)

model_zoo = sorted(name for name in models.__dict__
            if not name.startswith("__")
            and callable(models.__dict__[name]))


def parse_range(arg):
    param = map(float, arg.split(','))
    return np.arange(*param)


def parse_range_int(arg):
    param = map(int, arg.split(','))
    return np.arange(*param)


parser = argparse.ArgumentParser(description='Finetune parameters for SiamMask tracker on DAVIS')
parser.add_argument('--arch', dest='arch', default='SiamRPNA', choices=model_zoo + ['Custom'],
                    help='architecture of pretrained model')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config',help='hyperparameter of SiamMask in json format')
parser.add_argument('--mask', action='store_true', help='whether use mask output')
parser.add_argument('--refine', action='store_true', help='whether use mask refine output')
parser.add_argument('--dataset', default='DAVIS2016', type=str,
                    metavar='DATASET', help='dataset')
parser.add_argument('-l', '--log', default="log_tune_davis.txt", type=str,
                    help='log file')
parser.add_argument('--penalty-k', default='0.0,0.1,0.03', type=parse_range,
                    help='penalty_k range')
parser.add_argument('--lr', default='0.8,1.01,0.05', type=parse_range,
                    help='lr range')
parser.add_argument('--window-influence', default='0.3,0.5,0.04', type=parse_range,
                    help='window influence range')
parser.add_argument('--search-region', default='255,256,8', type=parse_range_int,
                    help='search region size')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')

args = parser.parse_args()


def tune(param):
    regions = []  # result and states[1 init / 2 lost / 0 skip]
    # save result
    benchmark_result_path = join('result', param['dataset'])
    tracker_path = join(benchmark_result_path, (param['network_name'] + ('_refine' if args.refine else '') +
                                                '_r{}'.format(param['hp']['instance_size']) +
                                                '_penalty_k_{:.3f}'.format(param['hp']['penalty_k']) +
                                                '_window_influence_{:.3f}'.format(param['hp']['window_influence']) +
                                                '_lr_{:.3f}'.format(param['hp']['lr'])).replace('.', '_'))  # no .
    video_path = tracker_path
    result_path = join(video_path, param['video'] + '.txt')

    if isfile(result_path):
        return

    try:
        if not isdir(video_path):
            makedirs(video_path)
    except OSError as err:
        print(err)

    with open(result_path, 'w') as f:  # Occupation
        f.write('Occ')
    
    global ims, gt, annos, image_files, anno_files
    if ims is None:
        print(param['video'] + '  Only load image once and if needed')
        ims = [cv2.imread(x) for x in image_files]
        annos = [np.array(Image.open(x)) for x in anno_files]

    iou = IouMeter(thrs, len(ims) - 2)
    start_frame, end_frame, toc = 0, len(ims) - 1, 0
    for f, (im, anno) in enumerate(zip(ims, annos)):
        tic = cv2.getTickCount()
        if f == start_frame:  # init
            target_pos = np.array([gt[f, 0]+gt[f, 2]/2, gt[f, 1]+gt[f, 3]/2])
            target_sz = np.array([gt[f, 2], gt[f, 3]])
            state = siamese_init(im, target_pos, target_sz, param['network'], param['hp'])  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(gt[f])
        elif f > start_frame:  # tracking
            state = siamese_track(state, im, args.mask, args.refine)  # track
            location = state['ploygon'].flatten()
            mask = state['mask']

            regions.append(location)
        if start_frame < f < end_frame: iou.add(mask, anno)

        toc += cv2.getTickCount() - tic

        if args.visualization and f >= start_frame:  # visualization (skip lost frame)
            im_show = im.copy()
            if f == 0: cv2.destroyAllWindows()
            if len(gt[f]) == 8:
                cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            else:
                cv2.rectangle(im_show, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            if len(location) == 8:
                im_show[:,:,2] = mask*255 + (1-mask)*im_show[:,:,2]
                cv2.polylines(im_show, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]  # bad support for OPENCV
                cv2.rectangle(im_show, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # frame id

            cv2.imshow(param['video'], im_show)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()
    iou_list = iou.value('mean')
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps IOU: {:.3f}'.format(param['video'], toc, f / toc, iou_list.max()))

    with open(result_path, 'w') as f:
        f.write(','.join(["%.5f" % i for i in iou_list]) + '\n')

    return iou_list


def main():
    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)
    
    params = {'penalty_k': args.penalty_k,
              'window_influence': args.window_influence,
              'lr': args.lr,
              'instance_size': args.search_region}

    num_search = len(params['penalty_k']) * len(params['window_influence']) * \
        len(params['lr']) * len(params['instance_size'])

    print(params)
    print(num_search)

    cfg = load_config(args)
    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(anchors=cfg['anchors'])
    else:
        model = models.__dict__[args.arch](anchors=cfg['anchors'])

    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model = load_pretrain(model, args.resume)
    model.eval()
    model = model.cuda()

    default_hp = cfg.get('hp', {})

    p = dict()

    p['network'] = model
    p['network_name'] = args.arch+'_mask_'+args.resume.split('/')[-1].split('.')[0]
    p['dataset'] = args.dataset

    global ims, gt, annos, image_files, anno_files

    dataset_info = load_dataset(args.dataset)
    videos = list(dataset_info.keys())
    np.random.shuffle(videos)
    for video in videos:
        print(video)
        p['video'] = video
        ims = None
        annos = None
        image_files = dataset_info[video]['image_files']
        anno_files = dataset_info[video]['anno_files']
        gt = dataset_info[video]['gt']

        np.random.shuffle(params['penalty_k'])
        np.random.shuffle(params['window_influence'])
        np.random.shuffle(params['lr'])
        for penalty_k in params['penalty_k']:
            for window_influence in params['window_influence']:
                for lr in params['lr']:
                    for instance_size in params['instance_size']:
                        p['hp'] = default_hp.copy()
                        p['hp'].update({'penalty_k':penalty_k,
                                'window_influence':window_influence,
                                'lr':lr,
                                'instance_size': instance_size,
                                        })
                        tune(p)


if __name__ == '__main__':
    main()
