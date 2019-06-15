import torch
import torch.optim as optim
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader

from SE_audio_torch import GRUAudio
from process_audio_torch import IEMOCAP, my_collate

import pdb
from tqdm import tqdm

# Detect the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize our GRU model with 39 features, hidden_dim=200, num_layers=2, droupout=0.7, num_labels=5
model = GRUAudio(num_features=39, hidden_dim=200, num_layers=2, dropout_rate=0.7, num_labels=5, batch_size=128)
model.cuda()
#pdb.set_trace()

# Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Load the training data
training_data = IEMOCAP(train=True)
train_loader = DataLoader(dataset=training_data, batch_size=128, shuffle=True, collate_fn=my_collate, num_workers=0)

# # See what the scores are before training
# # Note that element i,j of the output is the score for tag j for word i.
# # Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)

# Perform 10 epochs
for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    print("===================================" + str(epoch) + "==============================================")
    for j, (input, target, seq_length) in enumerate(train_loader):
        print("==============================Batch " + str(j) + "=============================================")
        # pad input sequence to make all the same length
        #pdb.set_trace()

        input = pad_sequence(sequences=input, batch_first=True)

        # seq_length = seq_length.to(device)
        # make input a packed padded sequence
#        pdb.set_trace()
        input = pack_padded_sequence(input, lengths=seq_length, batch_first=True, enforce_sorted=False)


        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 3. Run our forward pass.
        out, loss = model(input, target)

        print("Loss:", loss)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss.backward()
        optimizer.step()

    print("End of Epoch Loss: ", loss)
    print(model.state_dict())

torch.save(model.state_dict(), '/scratch/speech/models/classification/classifier.pt')



# # See what the scores are after training
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     out, loss = model(input, target)
#
#     # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#     # for word i. The predicted tag is the maximum scoring tag.
#     # Here, we can see the predicted sequence below is 0 1 2 0 1
#     # since 0 is index of the maximum value of row 1,
#     # 1 is the index of maximum value of row 2, etc.
#     # Which is DET NOUN VERB DET NOUN, the correct sequence!
#     print(loss)
