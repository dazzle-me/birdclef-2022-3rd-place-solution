import numpy as np
import sklearn.metrics

import torch

class CompMetric():
    def __init__(self, cfg, input=None, output=None):
        params = cfg['metrics']['bird_metric']

        self.input = input if input else params['input']
        self.output = output if output else params['output']

        self.name = 'bird_metric'
        self.threshold = params['threshold']
        self.scored_birds = cfg['general']['scored_birds']
        self.reset()

    def update(self, preds, data):
        input = preds[self.input]
        target = data[self.output]
        input = input[:, self.scored_birds]
        target = target[:, self.scored_birds]
        
        class_pred = input >= self.threshold
        self.preds.extend(self._convert_to_numpy(class_pred))
        self.gts.extend(self._convert_to_numpy(target))

    @staticmethod
    def _convert_to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError(f"Unacceptable type, expected one of (list, numpy.array, torch.Tensor), got {type(data)}")

    def compute(self):
        self.gts = np.array(self.gts)
        self.preds = np.array(self.preds)
        return self.comp_metric(self.gts, self.preds)

    @staticmethod
    def comp_metric(y_true, y_pred, epsilon=1e-9):
        """ Function to calculate competition metric in an sklearn like fashion

        Args:
            y_true{array-like, sparse matrix} of shape (n_samples, n_outputs)
                - Ground truth (correct) target values.
            y_pred{array-like, sparse matrix} of shape (n_samples, n_outputs)
                - Estimated targets as returned by a classifier.
        Returns:
            The single calculated score representative of this competitions evaluation
        """

        # Get representative confusion matrices for each label
        mlbl_cms = sklearn.metrics.multilabel_confusion_matrix(y_true, y_pred)

        # Get two scores (TP and TN SCORES)
        tp_scores = np.array([
            mlbl_cm[1, 1]/(epsilon+mlbl_cm[:, 1].sum()) \
            for mlbl_cm in mlbl_cms
            ])
        tn_scores = np.array([
            mlbl_cm[0, 0]/(epsilon+mlbl_cm[:, 0].sum()) \
            for mlbl_cm in mlbl_cms
            ])

        # Get average
        tp_mean = tp_scores.mean()
        tn_mean = tn_scores.mean()

        return round((tp_mean+tn_mean)/2, 8)
        
    def reset(self):
        self.preds = []
        self.gts = []