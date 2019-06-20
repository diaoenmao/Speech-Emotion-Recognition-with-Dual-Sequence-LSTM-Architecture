import torch
import torch.optim as optim
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader
from  torch.optim.lr_scheduler import CosineAnnealingLR as cos
from andre import  ATT
from process_audio_torch import IEMOCAP, my_collate

import pdb
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau




# Detect the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ATT(num_features=39, hidden_dim=300, num_layers=2, dropout_rate=0.0, num_labels=4, batch_size=128,bidirectional=True)
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.001)

training_data = IEMOCAP(train=True)
train_loader = DataLoader(dataset=training_data, batch_size=128, shuffle=True, collate_fn=my_collate, num_workers=0)

print(model.parameters)
#scheduler=cos(optimizer, 100)
scheduler=ReduceLROnPlateau(optimizer,factor=0.5,patience=4)
loss_summary=[]
f=open('/scratch/speech/models/classification/att_classifier.txt',"w+")
# Perform 10 epochs
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    print("===================================" + str(epoch) + "==============================================")
    losses = 0
    for j, (input, target, seq_length) in enumerate(train_loader):
#        print("==============================Batch " + str(j) + "=============================================")

        input = pad_sequence(sequences=input, batch_first=True)

        input = pack_padded_sequence(input, lengths=seq_length, batch_first=True, enforce_sorted=False)

        model.zero_grad()

        # Step 3. Run our forward pass.
        out, loss = model(input, target,seq_length)
        
        losses += loss.item() * target.shape[0]
        #print("Loss:", loss)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss.backward()
        optimizer.step()

    print("End of Epoch Mean Loss: ", losses / len(training_data))
    
    scheduler.step(losses/len(training_data))

    loss_summary.append((epoch,losses))
    f.write(str(epoch)+" : "+ str(losses/len(training_data))+"\n")
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), '/scratch/speech/models/classification/deep_att_classifier_epoch_' + str(epoch+1) + '.pt')
        print('/scratch/speech/models/classification/deep_att_classifier_epoch_' + str(epoch+1) + '.pt')

torch.save(model.state_dict(), '/scratch/speech/models/classification/deep_att_classifier.pt')

#f.write(" ".join(loss_summary))
f.close()
