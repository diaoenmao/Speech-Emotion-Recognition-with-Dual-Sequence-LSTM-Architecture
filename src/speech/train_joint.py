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
hidden_channels=[64,128,32]
kernel_size=[(3,3),(3,3),(3,3)]
kernel_size_pool=[(3,3),(3,3),(3,2)]
kernel_stride_pool=[(2,2),(2,2),(3,2)]
step=40
batch_size=80
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
optimizer = optim.Adam(model.parameters(),lr=0.0001)
optimizer2=optim.SGD(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=0.5, patience=2, threshold=1e-3)
#scheduler2=ReduceLROnPlateau(optimizer=optimizer2, factor=0.5, patience=2, threshold=1e-3)
scheduler2 =CosineAnnealingLR(optimizer2, T_max=100, eta_min=0.0001)


# Load the training data
training_data = IEMOCAP(train=True)
train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=0, drop_last=True)
testing_data = IEMOCAP(train=False)
test_loader = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=0,drop_last=True)

out = open('/scratch/speech/hand_raw_dataset/IEMOCAP_39_FOUR_EMO_spectrogram_segmented_dpi10_step40_overlap_test.pkl', 'rb')
data = pickle.load(out)
labels = np.array(data['target'])
weights=np.sum(labels,axis=0)/len(labels)
print("=================")
print("training data size: ", len(training_data))
with np.printoptions(precision=4, suppress=True):
    print("weights: ", weights)
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
        optimizer2.step()
        correct += correct_batch.item()
    accuracy=correct*1.0/((j+1)*batch_size)
    losses=losses / ((j+1)*batch_size)

    losses_test = 0
    correct_test = 0
    losses_test_ce=0
    torch.save(model.module.state_dict(), "/scratch/speech/models/classification/joint_checkpoint_epoch_{}.pt".format(epoch+1))
    model.eval()
    output = []
    y_true = []
    y_pred = []
    scheduler2.step()
    with torch.no_grad():
        for j,(input_lstm,input, target,seq_length) in enumerate(test_loader):
            if (j+1)%10==0: print("=================================Test Batch"+ str(j+1)+ "===================================================")
            input_lstm = pad_sequence(sequences=input_lstm,batch_first=True)
            losses_batch,losses_batch_ce,correct_batch, (target_index, pred_index)= model(input_lstm,input, target,seq_length, train=False)
            output.append((target_index, pred_index))
            loss = torch.mean(losses_batch,dim=0)
            correct_batch=torch.sum(correct_batch,dim=0)
            losses_test += loss.item() * batch_size
            losses_test_ce+=torch.mean(losses_batch_ce,dim=0).item()*batch_size
            correct_test += correct_batch.item()

    for target_index, pred_index in output:
        y_true = y_true + target_index.tolist()
        y_pred = y_pred + pred_index.tolist()
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("how many correct:", correct_test)
    print("confusion matrix: ")
    with np.printoptions(precision=4, suppress=True):
        print(cm_normalized)
    accuracy_test = correct_test * 1.0 / ((j+1)*batch_size)
    weighted_accuracy_test = 0
    for i in range(4):
        weighted_accuracy_test += cm_normalized[i,i] * weights[i]
    losses_test = losses_test / ((j+1)*batch_size)
    losses_test_ce=losses_test_ce/((j+1)*batch_size)

    # data gathering
    test_acc.append(accuracy_test)
    weighted_acc.append(weighted_accuracy_test)
    train_acc.append(accuracy)
    test_loss.append(losses_test)
    train_loss.append(losses)
    print("Epoch: {}----Training Loss: {:05.4f}----Testing Loss: {:05.4f}----Training Acc: {:05.4f}----Testing Acc: {:05.4f}----Weighted Acc: {:05.4f}-------CE Loss: {:05.4f}".format(epoch+1,losses,losses_test, accuracy, accuracy_test, weighted_accuracy_test, losses_test_ce)+"\n")
    with open("/scratch/speech/models/classification/joint_stats.txt","a+") as f:
        if epoch==0: f.write("\n"+"============================== New Model ==================================="+"\n")
        f.write("\n"+"Epoch: {}----Training Loss: {:05.4f}----Testing Loss: {:05.4f}----Training Acc: {:05.4f}----Testing Acc: {:05.4f}----Weighted Acc: {:05.4f}-------CE Loss: {:05.4f}".format(epoch+1,losses,losses_test, accuracy, accuracy_test, weighted_accuracy_test,losses_test_ce)+"\n")
        f.write("confusion_matrix:"+"\n")
        np.savetxt(f,cm_normalized,delimiter=' ',fmt="%5.4f")


pickle_out=open("/scratch/speech/models/classification/joint_checkpoint_stats.pkl","wb")
pickle.dump({"test_acc":test_acc, "weighted_acc": weighted_acc, "train_acc": train_acc, "train_loss": train_loss,"test_loss":test_loss},pickle_out)
pickle_out.close()
