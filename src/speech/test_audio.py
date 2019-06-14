import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from SE_audio_torch import GRUAudio
from process_audio_torch import IEMOCAP, my_collate

model = GRUAudio(num_features=39, hidden_dim=200, num_layers=2, dropout_rate=0.7, num_labels=5)
model.load_state_dict(torch.load('/scratch/speech/models/classification/classifier.torch.pt'))
model.eval()

testing_data = IEMOCAP(train=True)
test_loader = DataLoader(dataset=testing_data, batch_size=1, shuffle=True, collate_fn=my_collate, num_workers=0)