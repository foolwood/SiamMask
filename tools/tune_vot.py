# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import argparse
import logging
import numpy as np
import cv2
import torch
from os import makedirs
from os.path import isfile, isdir, join

from utils.log_helper import init_log, add_file_handler
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.load_helper import load_pretrain
from utils.benchmark_helper import load_dataset

from tools.test import siamese_init, siamese_track
from utils.config_helper import load_config
from utils.pyvotkit.region import vot_overlap, vot_float2str

def parse_range(arg):
    param = map(float, arg.split(','))
    return np.arange(*param)


def parse_range_int(arg):
    param = map(int, arg.split(','))
    return np.arange(*param)


parser = argparse.ArgumentParser(description='Finetune parameters for SiamMask tracker on VOT')
parser.add_argument('--arch', dest='arch', default='Custom', choices=['Custom', ],
                    help='architecture of pretrained model')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config',help='hyperparameter of SiamRPN in json format')
parser.add_argument('--mask', action='store_true', help='whether use mask output')
parser.add_argument('--refine', action='store_true', help='whether use mask refine output')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--dataset', default='VOT2018', type=str,
                    metavar='DATASET', help='dataset')
parser.add_argument('-l', '--log', default="log_tune.txt", type=str,
                    help='log file')
parser.add_argument('--penalty-k', default='0.05,0.5,0.05', type=parse_range,
                    help='penalty_k range')
parser.add_argument('--lr', default='0.35,0.5,0.05', type=parse_range,
                    help='lr range')
parser.add_argument('--window-influence', default='0.1,0.8,0.05', type=parse_range,
                    help='window influence range')
parser.add_argument('--search-region', default='255,256,8', type=parse_range_int,
                    help='search region size')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tune(param):
    regions = []  # result and states[1 init / 2 lost / 0 skip]
    # save result
    benchmark_result_path = join('result', param['dataset'])
    tracker_path = join(benchmark_result_path, (param['network_name'] +
                                                '_r{}'.format(param['hp']['instance_size']) +
                                                '_penalty_k_{:.3f}'.format(param['hp']['penalty_k']) +
                                                '_window_influence_{:.3f}'.format(param['hp']['window_influence']) +
                                                '_lr_{:.3f}'.format(param['hp']['lr'])).replace('.', '_'))  # no .
    if param['dataset'].startswith('VOT'):
        baseline_path = join(tracker_path, 'baseline')
        video_path = join(baseline_path, param['video'])
        result_path = join(video_path, param['video'] + '_001.txt')
    elif param['dataset'].startswith('OTB') or param['dataset'].startswith('DAVIS'):
        video_path = tracker_path
        result_path = join(video_path, param['video']+'.txt')

    if isfile(result_path):
        return

    try:
        if not isdir(video_path):
            makedirs(video_path)
    except OSError as err:
        print(err)

    with open(result_path, 'w') as f:  # Occupation
        f.write('Occ')
    
    global ims, gt, image_files
    if ims is None:
        print(param['video'] + '  Only load image once and if needed')
        ims = [cv2.imread(x) for x in image_files]
    start_frame, lost_times, toc = 0, 0, 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, param['network'], param['hp'], device=device)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            if param['dataset'].startswith('VOT'):
                regions.append(1)
            elif param['dataset'].startswith('OTB') or param['dataset'].startswith('DAVIS'):
                regions.append(gt[f])
        elif f > start_frame:  # tracking
            state = siamese_track(state, im, args.mask, args.refine, device=device)
            if args.mask:
                location = state['ploygon'].flatten()
            else:
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            if param['dataset'].startswith('VOT'):
                if 'VOT' in args.dataset:
                    gt_polygon = ((gt[f][0], gt[f][1]), 
                                  (gt[f][2], gt[f][3]),
                                  (gt[f][4], gt[f][5]), 
                                  (gt[f][6], gt[f][7]))
                    if args.mask:
                        pred_polygon = ((location[0], location[1]), (location[2], location[3]),
                                        (location[4], location[5]), (location[6], location[7]))
                    else:
                        pred_polygon = ((location[0], location[1]),
                                        (location[0] + location[2], location[1]),
                                        (location[0] + location[2], location[1] + location[3]),
                                        (location[0], location[1] + location[3]))
                    b_overlap = vot_overlap(gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))
                else:
                    b_overlap = 1

                if b_overlap:  # continue to track
                    regions.append(location)
                else:  # lost
                    regions.append(2)
                    lost_times += 1
                    start_frame = f + 5  # skip 5 frames
            else:
                regions.append(location)
        else:  # skip
            regions.append(0)
        toc += cv2.getTickCount() - tic

        if args.visualization and f >= start_frame:  # visualization (skip lost frame)
            if f == 0: cv2.destroyAllWindows()
            if len(gt[f]) == 8:
                cv2.polylines(im, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            else:
                cv2.rectangle(im, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            if len(location) == 8:
                location = np.int0(location)
                cv2.polylines(im, [location.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]  # bad support for OPENCV
                cv2.rectangle(im, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # frame id
            cv2.putText(im, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # lost time

            cv2.imshow(param['video'], im)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(param['video'], toc, f / toc, lost_times))

    with open(result_path, 'w') as f:
        for x in regions:
            f.write('{:d}\n'.format(x)) if isinstance(x, int) else \
                    f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')


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
    model = model.to(device)

    default_hp = cfg.get('hp', {})

    p = dict()

    p['network'] = model
    p['network_name'] = args.arch+'_'+args.resume.split('/')[-1].split('.')[0]
    p['dataset'] = args.dataset

    global ims, gt, image_files

    dataset_info = load_dataset(args.dataset)
    videos = list(dataset_info.keys())
    np.random.shuffle(videos)

    for video in videos:
        print(video)
        if isfile('finish.flag'):
            return

        p['video'] = video
        ims = None
        image_files = dataset_info[video]['image_files']
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
    with open('finish.flag', 'w') as f:  # Occupation
        f.write('finish')
