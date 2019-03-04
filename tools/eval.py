# --------------------------------------------------------
# Python Single Object Tracking Evaluation
# Licensed under The MIT License [see LICENSE for details]
# Written by Fangyi Zhang
# @author fangyi.zhang@vipl.ict.ac.cn
# @project https://github.com/StrangerZhang/pysot-toolkit.git
# Revised for SiamMask by foolwood
# --------------------------------------------------------
import argparse
import glob
from os.path import join, realpath, dirname

from tqdm import tqdm
from multiprocessing import Pool
from utils.pysot.datasets import VOTDataset
from utils.pysot.evaluation import AccuracyRobustnessBenchmark, EAOBenchmark

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VOT Evaluation')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--result_dir', type=str, help='tracker result root')
    parser.add_argument('--tracker_prefix', type=str, help='tracker prefix')
    parser.add_argument('--show_video_level', action='store_true')
    parser.add_argument('--num', type=int, help='number of processes to eval', default=1)
    args = parser.parse_args()

    root = join(realpath(dirname(__file__)), '../data')
    tracker_dir = args.result_dir
    trackers = glob.glob(join(tracker_dir, args.tracker_prefix+'*'))
    trackers = [t.split('/')[-1] for t in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'VOT2018' == args.dataset or 'VOT2016' == args.dataset:
        dataset = VOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                                                trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)

        benchmark = EAOBenchmark(dataset)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        ar_benchmark.show_result(ar_result, eao_result, show_video_level=args.show_video_level)
