import torch
import torch.optim as optim
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader

from attention import AttGRU
from process_audio_torch import IEMOCAP, my_collate

import pdb
from tqdm import tqdm

# Detect the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize our GRU model with 39 features, hidden_dim=200, num_layers=2, droupout=0.7, num_labels=5
model = AttGRU(num_features=39, hidden_dim=200, num_layers=2, dropout_rate=0.0, num_labels=5, batch_size=256)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

training_data = IEMOCAP(train=True)
train_loader = DataLoader(dataset=training_data, batch_size=256, shuffle=True, collate_fn=my_collate, num_workers=0)


# Perform 10 epochs
for epoch in range(5):  # again, normally you would NOT do 300 epochs, it is toy data
    print("===================================" + str(epoch) + "==============================================")
    losses = 0
    for j, (input, target, seq_length) in enumerate(train_loader):
#        print("==============================Batch " + str(j) + "=============================================")

        input = pad_sequence(sequences=input, batch_first=True)

        input = pack_padded_sequence(input, lengths=seq_length, batch_first=True, enforce_sorted=False)

        model.zero_grad()

        # Step 3. Run our forward pass.
        out, loss = model(input, target)
        
        losses += loss.item() * target.shape[0]
        #print("Loss:", loss)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss.backward()
        optimizer.step()

    print("End of Epoch Mean Loss: ", losses / len(training_data))
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), '/scratch/speech/models/classification/att_classifier_epoch_' + str(epoch+1) + '.pt')
    #print(model.state_dict())

torch.save(model.state_dict(), '/scratch/speech/models/classification/att_classifier.pt')
