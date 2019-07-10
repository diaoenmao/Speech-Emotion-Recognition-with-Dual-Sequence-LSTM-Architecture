import torch
from torch import optim
from model_joint import ConvLSTM
from process_joint import IEMOCAP,my_collate
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import pdb
from torch.nn import DataParallel
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_channels=3
hidden_channels=[64,32,16]
kernel_size=[(3,3),(3,3),(3,3)]
kernel_size_pool=[(3,3),(3,3),(3,2)]
kernel_stride_pool=[(2,2),(2,2),(3,2)]
step=40
batch_size=100
hidden_dim_lstm=200
num_layers_lstm=2
device_ids=[0,1,2,3]
num_devices=len(device_ids)
model = ConvLSTM(input_channels,hidden_channels,kernel_size,kernel_size_pool,kernel_stride_pool,step,device,num_devices,hidden_dim_lstm,num_layers_lstm)
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
training_data = IEMOCAP(train=True)
train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=0, drop_last=True)
testing_data = IEMOCAP(train=False)
test_loader = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=0,drop_last=True)

out = open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_test.pkl')
data = pickle.load(out)
labels = data['target']
hap_count = 0
neu_count = 0
ang_count = 0
sad_count = 0
for label in labels:
    if label == [1,0,0,0]:
        hap_count += 1
    elif label == [0,1,0,0]:
        neu_count += 1
    elif label == [0,0,1,0]:
        ang_count += 1
    else:
        sad_count += 1
weights = [hap_count/len(labels), neu_count/len(labels), ang_count/len(labels), sad_count/len(labels)]

print("=================")
print(len(training_data))
print("===================")
test_acc=[]
train_acc=[]
weighted_acc = []
test_loss=[]
train_loss=[]
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    print("===================================" + str(epoch+1) + "==============================================")
    losses = 0
    correct=0
    model.train()
    for j, (input_lstm,input, target,seq_length) in enumerate(train_loader):
        if (j+1)%20==0: print("=================================Train Batch"+ str(j+1)+ str(weight)+"===================================================")
        model.zero_grad()
        input_lstm = pad_sequence(sequences=input_lstm,batch_first=True)
        losses_batch,correct_batch= model(input_lstm,input, target,seq_length)
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
    torch.save(model.module.state_dict(), "/scratch/speech/models/classification/joint_checkpoint_epoch_{}.pt".format(epoch+1))
    model.eval()
    output = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for j,(input_lstm,input, target,seq_length) in enumerate(test_loader):
            if (j+1)%10==0: print("=================================Test Batch"+ str(j+1)+ "===================================================")
            input_lstm = pad_sequence(sequences=input_lstm,batch_first=True)
            losses_batch,correct_batch, (target_index, pred_index)= model(input_lstm,input, target,seq_length, train=False)
            output.append((target_index, pred_index))
            loss = torch.mean(losses_batch,dim=0)
            correct_batch=torch.sum(correct_batch,dim=0)
            losses_test += loss.item() * batch_size
            correct_test += correct_batch.item()

    for target_index, pred_index in output:
        y_true = y_true + target_index.tolist()
        y_pred = y_pred + pred_index.tolist()
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("how many correct:", correct_test)
    print("confusion matrix: ", cm)
    accuracy_test = correct_test * 1.0 / ((j+1)*batch_size)
    weighted_accuracy_test = 0
    for i in range(4):
        weighted_accuracy_test += cm_normalized[i,i] * weights[i]
    losses_test = losses_test / ((j+1)*batch_size)

    # data gathering
    test_acc.append(accuracy_test)
    weighted_acc.append(weighted_accuracy_test)
    train_acc.append(accuracy)
    test_loss.append(losses_test)
    train_loss.append(losses)
    print("Epoch: {}----Training Loss: {}----Testing Loss: {}----Training Acc: {}----Testing Acc: {}----Weighted Acc: {}".format(epoch+1,losses,losses_test, accuracy, accuracy_test, weighted_accuracy_test)+"\n")
    with open("/scratch/speech/models/classification/joint_stats.txt","a+") as f:
        if epoch==0: f.write("\n"+"============================== New Model ==================================="+"\n")
        f.write("Epoch: {}----Training Loss: {}----Testing Loss: {}----Training Acc: {}----Testing Acc: {}----Weighted Acc: {}".format(epoch+1,losses,losses_test, accuracy, accuracy_test, weighted_accuracy_test)+"\n")



pickle_out=open("/scratch/speech/models/classification/joint_checkpoint_stats.pkl","wb")
pickle.dump({"test_acc":test_acc, "weighted_acc": weighted_acc, "train_acc": train_acc, "train_loss": train_loss,"test_loss":test_loss},pickle_out)
pickle_out.close()
