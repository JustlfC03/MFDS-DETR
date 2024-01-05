import argparse
import pickle, json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import itertools
from terminaltables import AsciiTable

parser = argparse.ArgumentParser(description='Calculating metrics (AR, Recall) in  every class')
parser.add_argument('--det_json', default='results.json', type=str,
                    help='inference detection json file path')
parser.add_argument('--gt_json', default='/root/autodl-tmp/ZJHospital/annotations/instances_val2017.json', type=str,
                    help='ground truth json file path')
parser.add_argument('--classes', default=('Neutrophil', 'Monocyte', 'Eosinophil', 'Lymphocyte', 'Basophil'), type=tuple,
                    help='every class name with str type in a tuple')


def read_pickle(pkl):
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    return data


def read_json(json_pth):
    with open(json_pth, 'r') as f:
        data = json.load(f)
    return data


def process(det_json, gt_json, CLASSES):
    cocoGt = COCO(gt_json)

    # 获取类别（单类）对应的所有图片id
    catIds = cocoGt.getCatIds(catNms=list(CLASSES))  # ,'long','meihua'

    # 获取多个类别对应的所有图片的id
    imgid_list = []
    for id_c in catIds:
        imgIds = cocoGt.getImgIds(catIds=id_c)
        imgid_list.extend(imgIds)

    # 通过gt的json文件和pred的json文件计算map
    class_num = len(CLASSES)
    cocoGt = COCO(gt_json)
    cocoDt = cocoGt.loadRes(det_json)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    cocoEval.params.maxDets = list((100, 300, 1000))
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # 过gt和pred计算每个类别的recall
    precisions = cocoEval.eval['precision']  # TP/(TP+FP) right/detection
    recalls = cocoEval.eval['recall']  # iou*class_num*Areas*Max_det TP/(TP+FN) right/gt
    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[0], np.mean(precisions[0, :, :, 0, -1]),
                                                     np.mean(recalls[0, :, 0, -1])))
    # Compute per-category AP
    # from https://github.com/facebookresearch/detectron2/
    # precision: (iou, recall, cls, area range, max dets)
    results_per_category = []
    results_per_category_iou50 = []
    res_item = []
    for idx, catId in enumerate(range(class_num)):
        name = CLASSES[idx]
        precision = precisions[:, :, idx, 0, -1]
        precision_50 = precisions[0, :, idx, 0, -1]
        precision = precision[precision > -1]

        recall = recalls[:, idx, 0, -1]
        recall_50 = recalls[0, idx, 0, -1]
        recall = recall[recall > -1]

        if precision.size:
            ap = np.mean(precision)
            ap_50 = np.mean(precision_50)
            rec = np.mean(recall)
            rec_50 = np.mean(recall_50)
        else:
            ap = float('nan')
            ap_50 = float('nan')
            rec = float('nan')
            rec_50 = float('nan')
        res_item = [f'{name}', f'{float(ap):0.3f}', f'{float(rec):0.3f}']
        results_per_category.append(res_item)
        res_item_50 = [f'{name}', f'{float(ap_50):0.3f}', f'{float(rec_50):0.3f}']
        results_per_category_iou50.append(res_item_50)

    item_num = len(res_item)
    num_columns = min(6, len(results_per_category) * item_num)
    results_flatten = list(
        itertools.chain(*results_per_category))
    headers = ['category', 'AP', 'Recall'] * (num_columns // item_num)
    results_2d = itertools.zip_longest(*[
        results_flatten[i::num_columns]
        for i in range(num_columns)
    ])
    table_data = [headers]
    table_data += [result for result in results_2d]
    table = AsciiTable(table_data)
    print('\n' + table.table)

    num_columns_50 = min(6, len(results_per_category_iou50) * item_num)
    results_flatten_50 = list(
        itertools.chain(*results_per_category_iou50))
    iou_ = cocoEval.params.iouThrs[0]
    headers_50 = ['category', 'AP{}'.format(iou_), 'Recall{}'.format(iou_)] * (num_columns_50 // item_num)
    results_2d_50 = itertools.zip_longest(*[
        results_flatten_50[i::num_columns_50]
        for i in range(num_columns_50)
    ])

    table_data_50 = [headers_50]
    table_data_50 += [result for result in results_2d_50]
    table_50 = AsciiTable(table_data_50)
    print('\n' + table_50.table)


if __name__ == '__main__':
    args = parser.parse_args()
    process(det_json=args.det_json, gt_json=args.gt_json, CLASSES=args.classes)