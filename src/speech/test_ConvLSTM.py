import torch
from torch import optim
from raw_audio_model import RawAudioModel
from ConvLSTM import ConvLSTM
from process_raw_audio_model import IEMOCAP, my_collate
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import pdb
from torch.nn import DataParallel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch=1
model = torch.load("/scratch/speech/models/classification/ConvLSTM_checkpoint_epoch_{}.pt".format(epoch))
model=DataParallel(model,device_ids=[0,1,2,3])

# Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
optimizer = optim.SGD(model.parameters(), lr=0.01)
#scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=0.3, patience=5, threshold=1e-3)
scheduler =CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
# Load the training data
training_data = IEMOCAP(train=True)
train_loader = DataLoader(dataset=training_data, batch_size=60, shuffle=True, collate_fn=my_collate, num_workers=0)
testing_data = IEMOCAP(train=False)
test_loader = DataLoader(dataset=testing_data, batch_size=60, shuffle=True, collate_fn=my_collate, num_workers=0)

test_acc=[]
train_acc=[]
test_loss=[]
train_loss=[]

model.eval()
for test_case, target, seq_length in test_loader:
    test_case=test_case.float()
    test_case = test_case.unsqueeze(1)
    test_case=torch.split(test_case,1280,dim=2)
    try:
        out, loss = model(test_case, target, train=False, seq_length=seq_length)
    except:
        print(len(test_case))
        print(test_case[0].shape)
        print(target.shape)

    loss = torch.mean(loss,dim=0)
    out=torch.flatten(out,start_dim=0,end_dim=1)

    index = torch.argmax(out, dim=1)
    target_index = torch.argmax(target, dim=1).to(device)
    loss = torch.mean(loss)
    losses_test += loss.item() * index.shape[0]
    correct_test += sum(index == target_index).item()
accuracy_test = correct_test * 1.0 / len(testing_data)
losses_test = losses_test / len(testing_data)