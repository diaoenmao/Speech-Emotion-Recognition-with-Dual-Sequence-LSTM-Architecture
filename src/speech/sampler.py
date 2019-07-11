from torch.utils.data.sampler import Sampler
from process_joint_spec import IEMOCAP,my_collate
from torch.utils.data import DataLoader
import random
class SegmentCountSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        y = [(i, len(x)) for i, x in enumerate(self.data_source.seq_length_time)]
        #y = y.sort(key = lambda x: x[1])
        small = [], medium = [], large = [], extra = []
        for i, l in y:
            if l in range(1,3):
                small.append(i)
            elif l in range(3,6):
                medium.append(i)
            elif l in range(6,10):
                large.append(i)
            else:
                extra.append(i)
        small=random.shuffle(small)
        medium=random.shuffle(medium)
        large=random.shuffle(large)
        extra=random.shuffle(extra)
        print("shuffle")

        return small + medium + large + extra

    def __len__(self):
        return len(self.data_source)
if __name__=="__main__":
    name="mel"
    training_data = IEMOCAP(name=name,train=True)
    sampler_train=SegmentCountSampler(training_data)
    train_loader = DataLoader(dataset=training_data, batch_size=60, collate_fn=my_collate, num_workers=0, drop_last=True, sampler=sampler_train)
    for _ in range(10):
        for i,j in enumerate(train_loader):
            pass

