import numpy as np


class DenseMaskGenerator:
    def __init__(self):
        self.prune_threshold = 0
        self.add_threshold = 0
        self.mask = None
        self.count = 0

    def generate_mask(self, x, prune_threshold, add_threshold):
        self.prune_threshold = prune_threshold
        self.add_threshold = add_threshold
        sub_x = x.cpu().detach().numpy()
        self.mask = np.ones(sub_x)
        self.mask = np.where(np.abs(sub_x) < self.prune_threshold, 0, self.mask)
        self.mask = np.where(np.abs(sub_x) >= self.add_threshold, 1, self.mask)
        return self.mask

    def neuron_number(self, x):
        self.count = 0
        for i, j in enumerate(x):
            if np.all(self.mask.T[i] == 0):
                self.count += 1
        return self.count
