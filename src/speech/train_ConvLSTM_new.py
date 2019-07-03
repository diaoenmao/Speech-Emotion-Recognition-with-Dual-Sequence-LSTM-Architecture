import torch
from torch import optim
from raw_audio_model import RawAudioModel
from ConvLSTM import ConvLSTM
from process_raw_audio_model import IEMOCAP, my_collate_train, my_collate_test
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import pdb
from torch.nn import DataParallel
import pickle
import numpy as np
import torch.nn.functional as F

path="/scratch/speech/models/classification/ConvLSTM_data_debug.pickle"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_channels=[64,32,16]
kernel_size=[9,5,5]
step=100
device_ids=[0,1]
num_devices=len(device_ids)
model = ConvLSTM(1, hidden_channels,kernel_size,step,num_devices,True)
print("============================ Number of parameters ====================================")
print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model.cuda()
model=DataParallel(model,device_ids=device_ids)
model.train()

# Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer2=optim.SGD(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=0.5, patience=2, threshold=1e-3)
#scheduler2=ReduceLROnPlateau(optimizer=optimizer2, factor=0.5, patience=2, threshold=1e-3)
scheduler2 =CosineAnnealingLR(optimizer2, T_max=300, eta_min=0.0001)


# Load the training data, both use collate_test
training_data = IEMOCAP(train=True, segment=True)
train_loader = DataLoader(dataset=training_data, batch_size=16, shuffle=True, collate_fn=my_collate_test, num_workers=0, drop_last=True)
testing_data = IEMOCAP(train=False, segment=True)
test_loader = DataLoader(dataset=testing_data, batch_size=16, shuffle=True, collate_fn=my_collate_test, num_workers=0,drop_last=True)
print("=================")
print(len(training_data))
print("===================")
test_acc=[]
train_acc=[]
test_loss=[]
train_loss=[]
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    print("===================================" + str(epoch+1) + "==============================================")
    losses = 0
    correct=0
    losses_test = 0
    correct_test = 0
    length_full=0
    model.train()
    for j, (input, target, seq_length, segment_labels) in enumerate(train_loader):
        print(seq_length)
        model.zero_grad()
        losses_batch,correct_batch, length= model(input, target,seq_length)
        pdb.set_trace()
        loss=torch.mean(losses_batch,dim=0)
        length=torch.sum(length,dim=0)
        losses+=(loss*length).item()
        correct+=(torch.sum(correct_batch,dim=0)*length).item()
        loss.backward()
        optimizer.step()
        length_full+=length.item()
        if (j+1)%1==0: print("========================= Batch"+ str(j+1)+ str(length)+"=====================================")
    accuracy=correct*1.0/(len(training_data))
    losses=losses / (length_full)

    model.eval()
    with torch.no_grad():
        for test_case, target, seq_length,segment_labels in test_loader:
            losses_batch,correct_batch, length = model(test_case, target,seq_length, length)
            correct_test+=(torch.sum(correct_batch,dim=0)*torch.sum(length,dim=0)).item()

    accuracy_test = correct_test * 1.0 / (len(testing_data))
    #if losses_test<0.95: scheduler=scheduler2; optimizer=optimizer2

    # data gathering
    test_acc.append(accuracy_test)
    train_acc.append(accuracy)
    train_loss.append(losses)
    print("Epoch: {}-----------Training Loss: {}  -------- Training Acc: {} -------- Testing Acc: {}".format(epoch+1,losses, accuracy, accuracy_test)+"\n")
    with open("/scratch/speech/models/classification/ConvLSTM_checkpoint_stats.txt","a+") as f:
        if epoch==0: f.write("+\n"+"=============================  Begining New Model ======================================================="+"\n")
        f.write("Epoch: {}-----------Training Loss: {} -------- Training Acc: {} -------- Testing Acc: {}".format(epoch+1,losses, accuracy, accuracy_test)+"\n")
    scheduler.step(losses)
    #scheduler2.step()


pickle_out=open("/scratch/speech/models/classification/ConvLSTM_checkpoint_stats.pkl","wb")
pickle.dump({"test_acc":test_acc, "train_acc": train_acc, "train_loss": train_loss},pickle_out)
pickle_out.close()