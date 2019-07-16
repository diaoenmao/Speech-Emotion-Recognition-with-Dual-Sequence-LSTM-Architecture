import torch
from torch import optim
from model_joint_spec_full import SpectrogramModel
from process_joint_spec_full import IEMOCAP,my_collate
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import pdb
from torch.nn import DataParallel
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids=[0,1,2,3]
batch_size=80
input_channels=1
out_channels = [128,16]
kernel_size_cnn = [2,2]
stride_size_cnn = [1,1]
padding_cnn=[(1,0),(1,0)]
kernel_size_pool = [2,2]
stride_size_pool = [2,2]
hidden_dim=200
num_layers=2
dropout=0
num_labels=4
hidden_dim_lstm=250
num_layers_lstm=2
model = SpectrogramModel(input_channels,out_channels, kernel_size_cnn, stride_size_cnn, padding_cnn, kernel_size_pool, 
                            stride_size_pool, hidden_dim,num_layers,dropout,num_labels, batch_size, 
                            hidden_dim_lstm,num_layers_lstm,device,False)
print("============================ Number of parameters ====================================")
print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model.cuda()
model=DataParallel(model,device_ids=device_ids)
model.train()

# Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
optimizer = optim.Adam(model.parameters(),lr=0.001)
optimizer2=optim.SGD(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=0.5, patience=2, threshold=1e-3)
#scheduler2=ReduceLROnPlateau(optimizer=optimizer2, factor=0.5, patience=2, threshold=1e-3)
#scheduler2 =CosineAnnealingLR(optimizer2, T_max=300, eta_min=0.0001)


# Load the training data
training_data = IEMOCAP(name='mel', nfft=1024, train=True)
train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=0, drop_last=True)
testing_data = IEMOCAP(name='mel', nfft=1024, train=False)
test_loader = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=0,drop_last=True)
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
    model.train()
    for j, (input_lstm,input, target,seq_length,seq_length_spec) in enumerate(train_loader):
        if (j+1)%20==0: print("=================================Train Batch"+ str(j+1)+str(weight)+"===================================================")
        model.zero_grad()
        #input_lstm = pad_sequence(sequences=input_lstm,batch_first=True)
        losses_batch,correct_batch= model(input_lstm,input, target,seq_length,seq_length_spec)
        loss = torch.mean(losses_batch,dim=0)
        correct_batch=torch.sum(correct_batch,dim=0)
        losses += loss.item() * batch_size
        loss.backward()
        weight=model.module.state_dict()["weight"]
        weight=torch.exp(10*weight)/(1+torch.exp(10*weight)).item()
        optimizer.step()
        correct += correct_batch.item()
    accuracy=correct*1.0/((j+1)*batch_size)
    losses=losses / ((j+1)*batch_size)

    losses_test = 0
    correct_test = 0
    #torch.save(model.module.state_dict(), "/scratch/speech/models/classification/spec_full_joint_checkpoint_epoch_{}.pt".format(epoch+1))
    model.eval()
    with torch.no_grad():
        for j,(input_lstm,input, target,seq_length,seq_length_spec) in enumerate(test_loader):
            if (j+1)%10==0: print("=================================Test Batch"+ str(j+1)+ "===================================================")
            #input_lstm = pad_sequence(sequences=input_lstm,batch_first=True)
            losses_batch,correct_batch= model(input_lstm,input, target,seq_length,seq_length_spec)
            loss = torch.mean(losses_batch,dim=0)
            correct_batch=torch.sum(correct_batch,dim=0)
            losses_test += loss.item() * batch_size
            correct_test += correct_batch.item()

    print("how many correct:", correct_test)
    accuracy_test = correct_test * 1.0 / ((j+1)*batch_size)
    losses_test = losses_test / ((j+1)*batch_size)

    # data gathering
    test_acc.append(accuracy_test)
    train_acc.append(accuracy)
    test_loss.append(losses_test)
    train_loss.append(losses)
    print("Epoch: {}-----------Training Loss: {} -------- Testing Loss: {} -------- Training Acc: {} -------- Testing Acc: {}".format(epoch+1,losses,losses_test, accuracy, accuracy_test)+"\n")
    with open("/scratch/speech/models/classification/spec_full_joint_stats_2.txt","a+") as f:
        if epoch==0: f.write("\n"+"============================== New Model ==================================="+"\n")
        f.write("Epoch: {}-----------Training Loss: {} -------- Testing Loss: {} -------- Training Acc: {} -------- Testing Acc: {}".format(epoch+1,losses,losses_test, accuracy, accuracy_test)+"\n")



pickle_out=open("/scratch/speech/models/classification/spec_full_joint_2_checkpoint_stats.pkl","wb")
pickle.dump({"test_acc":test_acc, "train_acc": train_acc, "train_loss": train_loss,"test_loss":test_loss},pickle_out)
pickle_out.close()
