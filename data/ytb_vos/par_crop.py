# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import json
import glob


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_like_SiamFCx(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return x


def crop_video(video, v, crop_path, data_path, instanc_size):
    video_crop_base_path = join(crop_path, video)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)

    anno_base_path = join(data_path, 'Annotations')
    img_base_path = join(data_path, 'JPEGImages')

    for trackid, o in enumerate(list(v)):
        obj = v[o]
        for frame in obj:
            file_name = frame['file_name']
            ann_path = join(anno_base_path, file_name+'.png')
            img_path = join(img_base_path, file_name+'.jpg')
            im = cv2.imread(img_path)
            label = cv2.imread(ann_path, 0)
            avg_chans = np.mean(im, axis=(0, 1))
            bbox = frame['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            x = crop_like_SiamFCx(im, bbox, instanc_size=instanc_size, padding=avg_chans)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(file_name.split('/')[-1]), trackid)), x)
            mask = crop_like_SiamFCx((label==int(o)).astype(np.float32), bbox, instanc_size=instanc_size, padding=0)
            mask = ((mask > 0.2)*255).astype(np.uint8)
            x[:,:,0] = mask + (mask == 0)*x[:,:,0]
            # cv2.imshow('maskonx', x)
            # cv2.waitKey(0)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.m.png'.format(int(file_name.split('/')[-1]), trackid)), mask)


def main(instanc_size=511, num_threads=12):
    dataDir = '.'
    crop_path = './crop{:d}'.format(instanc_size)
    if not isdir(crop_path): mkdir(crop_path)

    for dataType in ['train']:
        set_crop_base_path = join(crop_path, dataType)
        set_img_base_path = join(dataDir, dataType)

        annFile = '{}/instances_{}.json'.format(dataDir, dataType)
        ytb_vos = json.load(open(annFile,'r'))
        n_video = len(ytb_vos)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, k, v, set_crop_base_path, set_img_base_path, instanc_size)
                  for k,v in ytb_vos.items()]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_video, prefix=dataType, suffix='Done ', barLength=40)
    print('done')


if __name__ == '__main__':
    since = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
