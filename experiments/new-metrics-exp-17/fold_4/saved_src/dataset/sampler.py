from enum import unique
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random

import numpy as np
import os
import json
import pandas as pd

## Rewrite sampler to be
## independent from source
class BalancedSampler(Sampler):
    def __init__(self, cfg):
        self.cfg = cfg
    
        params = cfg['data']['sampler']
        
        batch_size = cfg['data']['batch_size']
        self.num_instances_per_unique_id = params['instances_per_id']
        self.unique_ids_per_batch = batch_size // self.num_instances_per_unique_id

        self.index_dict = defaultdict(list)
	
	#### rewrite this part
        self.data_root = cfg['data']['dir']
        self.image_dir = os.path.join(cfg['data']['dir'], 'train_images_cropped')

        self.metadata = pd.read_csv(os.path.join(self.data_root, cfg['data']['csv_file']))
        
        valid_idx = np.load(os.path.join(self.data_root, f"stage_{cfg['general']['stage']}_train.npy"))
        self.metadata = self.metadata.iloc[valid_idx].reset_index(drop=True)

        if self.cfg['general']['dev']:
            self.metadata = self.metadata[self.metadata.individual_id.apply(lambda x : x in range(15))]
            self.metadata = self.metadata[:5000]
        
        self.data_source = self.metadata.reset_index(drop=True)
	#### rewrite this part
	
        for index, row in enumerate(self.data_source.itertuples()):
            individual_id = row.individual_id
            self.index_dict[individual_id].append(index)
        
        self.unique_ids = list(self.index_dict.keys())
        print(f"len(self.unique_ids) : {len(self.unique_ids)}")
        ## estimate number of examples in epoch
        self.length = 0
        for unique_id in self.unique_ids:
            idxs = self.index_dict[unique_id]
            num = len(idxs)
            if num < self.num_instances_per_unique_id:
                num = self.num_instances_per_unique_id

            self.length += num - num % self.num_instances_per_unique_id

        print(self.length)
    
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for unique_id in self.unique_ids:
            idxs = copy.deepcopy(self.index_dict[unique_id])
            if len(idxs) < self.num_instances_per_unique_id:
                idxs = np.random.choice(idxs, size=self.num_instances_per_unique_id, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            ## now we have dict with collections
            ## of indicies for every unique_id of size :param: num_insatnces_per_unique_id
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances_per_unique_id:
                    batch_idxs_dict[unique_id].append(batch_idxs)
                    batch_idxs = []

        ## it's left to construct the list of indices to take from:
        ## in the nutshell we get the sequence of indices such that:
        ## for every batch_size indices we have:
        ## :param: unique_ids_per_batch - amount of unique ids per batch
        ## :param: num_instances_per_unique_id - amount of instance for given id
        available_ids = copy.deepcopy(self.unique_ids)
        final_idxs = []
        while len(available_ids) >= self.unique_ids_per_batch:
            selected_ids = random.sample(available_ids, self.unique_ids_per_batch)
            for unique_id in selected_ids:
                batch_idxs = batch_idxs_dict[unique_id].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[unique_id]) == 0:
                    available_ids.remove(unique_id)
        return iter(final_idxs)
    
    def __len__(self):
        return self.length
