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
path="/scratch/speech/models/classification/ConvLSTM_data_debug.pickle"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_channels=[64,32,16]
kernel_size=[9,5,5]
step=100
model = ConvLSTM(1, hidden_channels,kernel_size,step,True)
print("============================ Number of parameters ====================================")
print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model.cuda()
device_ids=[0,1,2,3]
num_devices=len(device_ids)
model=DataParallel(model,device_ids=device_ids)
model.train()

# Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer2=optim.SGD(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=0.5, patience=2, threshold=1e-3)
#scheduler2=ReduceLROnPlateau(optimizer=optimizer2, factor=0.5, patience=2, threshold=1e-3)
scheduler2 =CosineAnnealingLR(optimizer2, T_max=300, eta_min=0.0001)
# Load the training data
training_data = IEMOCAP(train=True, segment=True)
train_loader = DataLoader(dataset=training_data, batch_size=100, shuffle=True, collate_fn=my_collate_train, num_workers=0)
testing_data = IEMOCAP(train=False, segment=True)
test_loader = DataLoader(dataset=testing_data, batch_size=100, shuffle=True, collate_fn=my_collate_test, num_workers=0)
print("=================")
print(len(training_data))
print("===================")
test_acc=[]
train_acc=[]
test_loss=[]
train_loss=[]
epoch=0
torch.save(model.module.state_dict(), "/scratch/speech/models/classification/ConvLSTM_checkpoint_epoch_{}.pt".format(epoch))
losses_test = 0
correct_test = 0
model1=ConvLSTM(1, hidden_channels,kernel_size,step,True)
model1.load_state_dict(torch.load("/scratch/speech/models/classification/ConvLSTM_checkpoint_epoch_{}.pt".format(epoch)))
model1.eval()
with torch.cuda.device(1):
    model1.cuda()
print("success")
with torch.no_grad():
    for j,(test_case, target, seq_length) in enumerate(test_loader):
        print(j)
        temp=[]
        for i in test_case:
            for j in i:
                temp.append(j)
        test_case=torch.from_numpy(np.array([i for i in temp])).to(device)
        length=test_case.shape[0]

        test_case=test_case.float()
        test_case = test_case.unsqueeze(1)
        test_case=torch.split(test_case,int(32000/step),dim=2)
        pdb.set_trace()
        out = model(test_case, target)

        out=torch.flatten(out,start_dim=0,end_dim=1)


        target_index = torch.argmax(target, dim=1).to(device)

        temp=0
        temp1=0
        for i,j in enumerate(target_index):
            temp1+=seq_length[i]
            if j==torch.argmax(torch.sum(out[temp:temp1,:],dim=0)):
                correct_test+=1
        print(correct_test)
accuracy_test = correct_test * 1.0 / (len(testing_data)-res)
print(accuracy_test)






























torch.save(model.state_dict(), "/scratch/speech/models/classification/ConvLSTM_checkpoint_epoch_{}.pt".format(epoch))
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    print("===================================" + str(epoch+1) + "==============================================")
    losses = 0
    correct=0
    losses_test = 0
    correct_test = 0
    model.train()
    for j, (input, target, _) in enumerate(train_loader):
        if (j+1)%5==0: print("================================= Batch"+ str(j+1)+ "===================================================")
        input=input.float()
        input = input.unsqueeze(1)
        input=torch.split(input,int(32000/step),dim=2)
        res=target.shape[0]%num_devices
        quo=target.shape[0]//num_devices
        if res !=0:
            target=target[:num_devices*quo]
            input=[t[:num_devices*quo] for t in input]

        model.zero_grad()
        out, loss = model(input, target)
        #pdb.set_trace()
        loss = torch.mean(loss,dim=0)
        out=torch.flatten(out,start_dim=0,end_dim=1)
        #pdb.set_trace()
        losses += loss.item() * target.shape[0]
        loss.backward()
        optimizer.step()

        index = torch.argmax(out, dim=1)
        target_index = torch.argmax(target, dim=1).to(device)
        correct += sum(index == target_index).item()

    accuracy=correct*1.0/(len(training_data)-res)
    losses=losses / (len(training_data)-res)
    print("accuracy:", accuracy)
    print("loss:", losses)
    torch.save(model.state_dict(), "/scratch/speech/models/classification/ConvLSTM_checkpoint_epoch_{}.pt".format(epoch+1))

    model.eval()
    with torch.no_grad():
        for test_case, target, seq_length in test_loader:
            temp=[]
            for i in test_case:
                for j in i:
                    temp.append(i)
            test_case=temp
            test_case=torch.from_numpy(np.array([i for i in test_case])).to(device)

            test_case=test_case.float()
            test_case = test_case.unsqueeze(1)
            test_case=torch.split(test_case,int(32000/step),dim=2)


            res=target.shape[0]%num_devices
            quo=target.shape[0]//num_devices
            if res !=0:
                target=target[:num_devices*quo]
                test_case=[t[:num_devices*quo] for t in test_case]
            out, loss = model(test_case, target)

            loss = torch.flatten(loss,start_dim=0, end_dim=1)
            out=torch.flatten(out,start_dim=0,end_dim=1)


            target_index = torch.argmax(target, dim=1).to(device)

            temp=0
            temp1=0
            for i,j in enumerate(target_index):
                temp1+=seq_length[i]
                if j==torch.argmax(torch.sum(out[temp:temp1,:],dim=0)):
                    correct_test+=1
                losses_test+=torch.sum(out[temp:temp1,j]).item()
    accuracy_test = correct_test * 1.0 / (len(testing_data)-res)
    losses_test = losses_test / (len(testing_data)-res)
    #if losses_test<0.95: scheduler=scheduler2; optimizer=optimizer2

    # data gathering
    test_acc.append(accuracy_test)
    train_acc.append(accuracy)
    test_loss.append(losses_test)
    train_loss.append(losses)
    print("Epoch: {}-----------Training Loss: {} -------- Testing Loss: {} -------- Training Acc: {} -------- Testing Acc: {}".format(epoch+1,losses,losses_test, accuracy, accuracy_test)+"\n")
    with open("/scratch/speech/models/classification/ConvLSTM_checkpoint_stats.txt","a+") as f:
        f.write("Epoch: {}-----------Training Loss: {} -------- Testing Loss: {} -------- Training Acc: {} -------- Testing Acc: {}".format(epoch+1,losses,losses_test, accuracy, accuracy_test)+"\n")


    scheduler.step(losses)
    #scheduler2.step()


pickle_out=open("/scratch/speech/models/classification/ConvLSTM_checkpoint_stats.pkl","wb")
pickle.dump({"test_acc":test_acc, "train_acc": train_acc, "test_loss": test_loss, "train_loss": train_loss},pickle_out)
pickle_out.close()