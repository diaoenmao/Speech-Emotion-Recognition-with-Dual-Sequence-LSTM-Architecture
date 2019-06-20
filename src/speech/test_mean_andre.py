import argparse
import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader

from andre import ATT, Mean_Pool_2
from process_audio_torch import IEMOCAP, my_collate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in range(10):

    model = Mean_Pool_2(num_features=39, hidden_dim=300, num_layers=2, dropout_rate=0.2,
                     num_labels=4, batch_size=128,bidirectional=True)
    model = model.cuda()
    diri="/scratch/speech/models/classification/deep_mean_pool_epoch_"+str((i+1)*10)+".pt"
    model.load_state_dict(torch.load(diri))
    model.eval()

    testing_data = IEMOCAP(train=False)
    test_loader = DataLoader(dataset=testing_data, batch_size=256, shuffle=True, collate_fn=my_collate, num_workers=0)
    print("Loading successful")

    print(len(testing_data))
    losses = 0
    correct = 0
    for test_case, target, seq_length in test_loader:
        test_case = pad_sequence(sequences=test_case, batch_first=True)
        test_case = pack_padded_sequence(test_case, lengths=seq_length, batch_first=True, enforce_sorted=False)
        out, loss = model(test_case, target,seq_length, False)
        index = torch.argmax(out, dim=1)
        target_index = torch.argmax(target, dim=1).to(device)
        losses += loss.item() * index.shape[0]
        correct += sum(index == target_index).item()
    accuracy = correct * 1.0 / len(testing_data)
    losses = losses / len(testing_data)
    print("what epoch:", (i+1)*10)
    print("accuracy:", accuracy)
    print("loss:", losses)