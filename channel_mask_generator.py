import numpy as np


class ChannelMaskGenerator:
    def __init__(self):
        self.prune_threshold = 0
        self.add_threshold = 0
        self.mask = None
        self.count = 0

    def generate_mask(self, x, pre_x, prune_threshold, add_threshold):
        self.prune_threshold = prune_threshold
        sub_x = x.cpu().detach().numpy()
        self.mask = np.ones(sub_x.shape)
        if pre_x is not None:
            sub_pre_x = pre_x.cpu().detach().numpy()
            for i, j in enumerate(sub_pre_x):
                if np.sum(np.abs(sub_pre_x[i])) == 0:
                    for k, l in enumerate(self.mask):
                        self.mask[k][i] = 0
        for i, j in enumerate(sub_x):
            l1_norm = np.sum(np.abs(sub_x[i]))
            if l1_norm < self.prune_threshold:
                self.mask[i] = 0
        return self.mask

    def channel_number(self, x):
        self.count = 0
        sub_x = x.cpu().detach().numpy()
        for i, j in enumerate(sub_x):
            if np.all(self.mask[i] == 0):
                self.count += 1
        return self.count
