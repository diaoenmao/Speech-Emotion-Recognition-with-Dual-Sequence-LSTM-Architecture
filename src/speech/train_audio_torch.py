import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader

from SE_audio_torch import GRUAudio
from process_audio_torch import IEMOCAP, my_collate

model = GRUAudio(39, 200, 2, 0.7, 5)
#loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

training_data = IEMOCAP(True)
train_loader = DataLoader(dataset=training_data, batch_size=128, shuffle=True, collate_fn=my_collate, num_workers=0)

# # See what the scores are before training
# # Note that element i,j of the output is the score for tag j for word i.
# # Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)

for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    print("===================================" + str(epoch) + "==============================================")
    for input, target, seq_length in train_loader:

        input = pad_sequence(sequences=input,batch_first=True)

        input = pack_padded_sequence(input, lengths=seq_length, batch_first=True, enforce_sorted=False)

        target = torch.from_numpy(np.array(target))

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 3. Run our forward pass.
        out, loss = model(input, target)


        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss.backward()
        optimizer.step()

    print("End of Epoch Loss: ", loss)

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
