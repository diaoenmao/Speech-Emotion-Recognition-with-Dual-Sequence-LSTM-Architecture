import torch.utils.data.sampler import Sampler

class SegmentCountSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        y = [(i, len(x)) for i, x in enumerate(self.data_source.seq_length_time)]
        #y = y.sort(key = lambda x: x[1])
        small = [], medium = [], large = [], extra = []
        for i, len in y:
            if len in range(1,3):
                small.append(i)
            elif len in range(3,6):
                medium.append(i)
            elif len in range(6,10):
                large.append(i)
            else:
                extra.append(i)
        return small + medium + large + extra

    def __len__(self):
        return len(self.data_source)
