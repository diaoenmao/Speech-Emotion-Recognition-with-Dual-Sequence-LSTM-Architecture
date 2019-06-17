import pdb
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from SE_audio_torch import GRUAudio
from process_audio_torch import IEMOCAP, my_collate

model = GRUAudio(num_features=39, hidden_dim=200, num_layers=2, dropout_rate=0.7, num_labels=5, batch_size=1)
model = model.cuda()
model.load_state_dict(torch.load('/scratch/speech/models/classification/classifier_epoch_30.pt'))
model.eval()

testing_data = IEMOCAP(train=False)
test_loader = DataLoader(dataset=testing_data, batch_size=1, shuffle=True, collate_fn=my_collate, num_workers=0)
print("Loading successful")

correct = 0
print(len(test_loader))
for i, (test_case, target, _) in enumerate(test_loader):
    if i % 100 == 0:
        print(i)
    test_case = test_case[0]
    test_case= test_case.reshape(1,test_case.shape[0],test_case.shape[1])
    out, loss = model(test_case,target,False)
#    pdb.set_trace()
    index = torch.argmax(out)
#    print(index)
  #  pdb.set_trace()
#    print("sample:",i)
    if target[0][index] == 1:
 #       print("Success!!")
        correct += 1

accuracy = correct * 1.0 / len(testing_data)
print("accuracy:", accuracy)
