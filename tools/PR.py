# =========================================================
# @purpose: plot PR curve by COCO API and mmdet API
# @date：   2020/12
# @version: v1.0
# @author： Xu Huasheng
# @github： https://github.com/xuhuasheng/mmdetection_plot_pr_curve
# =========================================================

import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmcv import Config
from mmdet.datasets import build_dataset


CONFIG_FILE = '../tools/work_dirs/PSNR10-0.833/cascade_rcnn_r50_rfp_1x_coco.py'
RESULT_FILE ='louci_0.92.pkl'


def plot_pr_curve(config_file, result_file, metric="bbox"):
    """plot precison-recall curve based on testing results of pkl file.

        Args:
            config_file (list[list | tuple]): config file path.
            result_file (str): pkl file of testing results path.
            metric (str): Metrics to be evaluated. Options are
                'bbox', 'segm'.
    """

    cfg = Config.fromfile(config_file)
    # turn on test mode of dataset
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build dataset
    dataset = build_dataset(cfg.data.test)
    # load result file in pkl format
    pkl_results = mmcv.load(result_file)
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results, _ = dataset.format_results(pkl_results)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results[metric])
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    # coco_eval.params.catIds=[6]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.catIds = [1]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.catIds = [2]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.catIds = [3]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.catIds = [4]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.catIds = [5]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.catIds = [6]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.catIds = [7]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.catIds = [8]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.catIds = [9]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.params.catIds = [10]
    coco_eval.params.iouThrs = [0.5]
    # coco_eval.params.areaRngLbl=['small']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show()
    np.savetxt("20.txt",pr_array6)



if __name__ == "__main__":
    plot_pr_curve(config_file=CONFIG_FILE, result_file=RESULT_FILE, metric="bbox")







