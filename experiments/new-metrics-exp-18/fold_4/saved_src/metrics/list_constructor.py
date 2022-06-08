from metrics.accuracy import Accuracy
from metrics.map import mAP
from metrics.f1_score import F1
from metrics.bird_metric import CompMetric

from pprint import pprint

def list_constructor(cfg):
    metric_list = []
    pprint(cfg['metrics'])
    for metric_name, params in cfg['metrics'].items():
        if "f1" in metric_name:
            metric_list.append(F1(cfg, params))
    return metric_list