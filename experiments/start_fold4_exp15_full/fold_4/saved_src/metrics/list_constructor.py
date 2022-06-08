from metrics.accuracy import Accuracy
from metrics.map import mAP
from metrics.f1_score import F1
from metrics.bird_metric import CompMetric

def list_constructor(cfg):
    metric_names = cfg['metrics']['collect']
    metric_list = []
    if 'accuracy' in metric_names:
        metric_list.append(Accuracy(cfg))
    if 'mAP' in metric_names:
        metric_list.append(mAP(cfg))
    if 'f1' in metric_names:
        metric_list.append(F1(cfg))
    if 'bird_metric' in metric_names:
        metric_list.append(CompMetric(cfg))
    return metric_list