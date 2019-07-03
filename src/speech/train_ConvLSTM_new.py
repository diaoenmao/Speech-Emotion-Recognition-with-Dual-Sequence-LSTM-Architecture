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
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
hidden_channels=[64,32,16]
kernel_size=[9,5,5]
step=100
model = ConvLSTM(1, hidden_channels,kernel_size,step,True)
print("============================ Number of parameters ====================================")
print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
with torch.cuda.device(1):
    model.cuda()
model.train()

# Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer2=optim.SGD(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=0.5, patience=2, threshold=1e-3)
#scheduler2=ReduceLROnPlateau(optimizer=optimizer2, factor=0.5, patience=2, threshold=1e-3)
scheduler2 =CosineAnnealingLR(optimizer2, T_max=300, eta_min=0.0001)


# Load the training data, both use collate_test
training_data = IEMOCAP(train=True, segment=True)
train_loader = DataLoader(dataset=training_data, batch_size=100, shuffle=True, collate_fn=my_collate_test, num_workers=0, drop_last=True)
testing_data = IEMOCAP(train=False, segment=True)
test_loader = DataLoader(dataset=testing_data, batch_size=100, shuffle=True, collate_fn=my_collate_test, num_workers=0)
print("=================")
print(len(training_data))
print("===================")
test_acc=[]
train_acc=[]
test_loss=[]
train_loss=[]
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    print("===================================" + str(epoch+1) + "==============================================")
    losses = 0
    correct=0
    losses_test = 0
    correct_test = 0
    model.train()
    for j, (input, target, seq_length, segment_labels) in enumerate(train_loader):
        if (j+1)%5==0: print("================================= Batch"+ str(j+1)+ "===================================================")
        temp=[]
        for i in input:
            for k in i:
                temp.append(k)
        input=torch.from_numpy(np.array([i for i in temp])).to(device)
        length=input.shape[0]
        input=input.float()
        input = input.unsqueeze(1)
        input=torch.split(input,int(32000/step),dim=2)
        pdb.set_trace()
        temp=[]

        model.zero_grad()
        out,_ = model(input, target)
        target_index = torch.argmax(target, dim=1).to(device)
        temp=0
        temp1=0
        for i,j in enumerate(target_index):
            temp1+=seq_length[i].item()
            loss=torch.sum(out[temp:temp1,j],dim=0)
            if j==torch.argmax(torch.sum(out[temp:temp1,:],dim=0)):
                correct+=1
            temp=temp1
        losses += loss.item()
        losses_mean=losses/length
        losses.backward()
        optimizer.step()

    accuracy=correct*1.0/(len(training_data))
    losses=losses / (len(training_data))

    model.eval()
    with torch.no_grad():
        for test_case, target, seq_length,segment_labels in test_loader:
            temp=[]
            for i in test_case:
                for k in i:
                    temp.append(k)
            test_case=torch.from_numpy(np.array([i for i in temp])).to(device)
            length=test_case.shape[0]

            test_case=test_case.float()
            test_case = test_case.unsqueeze(1)
            test_case=torch.split(test_case,int(32000/step),dim=2)
            out,_ = model(test_case, target)
            target_index = torch.argmax(target, dim=1).to(device)
            temp=0
            temp1=0
            for i,j in enumerate(target_index):
                temp1+=seq_length[i].item()
                if j==torch.argmax(torch.sum(out[temp:temp1,:],dim=0)):
                    correct_test+=1
                temp=temp1

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