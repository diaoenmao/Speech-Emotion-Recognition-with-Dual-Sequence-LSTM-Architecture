import torch
from torch import optim
from model_joint_convLSTM_1d import ConvLSTM
from process_joint_convLSTM_1d import IEMOCAP, my_collate
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR
import pdb
from torch.nn import DataParallel
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import argparse

def init_parser():
    parser = argparse.ArgumentParser(description='Train and test your model as specified by the parameters you enter')
    parser.add_argument('--batch_size', '-b', default=128, type=int, dest='batch_size')
    parser.add_argument('--out_channels_1', '-out1', default=64, type=int, dest='out_channels1')
    parser.add_argument('--out_channels_2', '-out2', default=16, type=int, dest='out_channels2')
    parser.add_argument('--kernel_size_cnn_1', '-kc1', default=2, type=int, dest='kernel_size_cnn1')
    parser.add_argument('--stride_size_cnn_1', '-sc1', default=1, type=int, dest='stride_size_cnn1')
    parser.add_argument('--kernel_size_pool_1', '-kp1', default=2, type=int, dest='kernel_size_pool1')
    parser.add_argument('--stride_size_pool', '-sp', default=2, type=int, dest='stride_size_pool')
    return parser.parse_args()

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_ids=[0,1,2,3]
    batch_size=args.batch_size
    input_channels = 1
    hidden_channels = [args.out_channels1, args.out_channels2]
    kernel_size_cnn = [args.kernel_size_cnn1]*2
    stride_size_cnn = [args.stride_size_cnn1]*2
    kernel_size_pool = [args.kernel_size_pool1]*2
    stride_size_pool = [args.stride_size_pool]*2
    num_layers=2
    dropout=0
    num_labels=4
    hidden_dim_lstm=200
    epoch_num=50
    num_layers_lstm=2
    model = ConvLSTM(input_channels,hidden_channels,kernel_size_cnn,stride_size_cnn,kernel_size_pool,stride_size_pool,hidden_dim_lstm,num_layers_lstm,device)
    
    print("============================ Number of parameters ====================================")
    print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    path="batch_size:{};out_channels:{};kernel_size_cnn:{};stride_size_cnn:{};kernel_size_pool:{};stride_size_pool:{}".format(args.batch_size,hidden_channels,kernel_size_cnn,stride_size_cnn,kernel_size_pool,stride_size_pool)
    
    with open("/scratch/speech/models/classification/joint_convLSTM_1d.txt","a+") as f:
        f.write("\n"+"============ model starts ===========")
        f.write("\n"+"mnodel_parameters: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad))+"\n"+path+"\n")
    
    model.cuda()
    model=DataParallel(model,device_ids=device_ids)
    model.train()

    # Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    optimizer2=optim.SGD(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=0.5, patience=2, threshold=1e-3)
    #scheduler2=ReduceLROnPlateau(optimizer=optimizer2, factor=0.5, patience=2, threshold=1e-3)
    #scheduler2 =CosineAnnealingLR(optimizer2, T_max=300, eta_min=0.0001)
    scheduler3 =MultiStepLR(optimizer, [5,10,15,20,25,30,35,40,45],gamma=0.5)

    # Load the training data
    training_data = IEMOCAP(train=True)
    train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=0, drop_last=True)
    testing_data = IEMOCAP(train=False)
    test_loader = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=0,drop_last=True)
    
    print("=================")
    print(len(training_data))
    print("===================")
    

    test_acc=[]
    train_acc=[]
    test_loss=[]
    train_loss=[]
    print("Model Initialized: {}".format(path)+"\n")
    for epoch in range(epoch_num):  # again, normally you would NOT do 300 epochs, it is toy data
        print("===================================" + str(epoch+1) + "==============================================")
        losses = 0
        correct=0
        model.train()
        for j, (input_lstm, input, target, seq_length) in enumerate(train_loader):
            if (j+1)%5==0:
                print("=================================Train Batch "+ str(j+1)+": "+str(weight.item())+"===================================================")
            model.zero_grad()
            losses_batch,correct_batch= model(input_lstm, input, target, seq_length)
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
        #scheduler3.step()
        losses_test = 0
        correct_test = 0
        #torch.save(model.module.state_dict(), "/scratch/speech/models/classification/spec_full_joint_checkpoint_epoch_{}.pt".format(epoch+1))
        model.eval()
        with torch.no_grad():
            for j,(input_lstm, input, target, seq_length) in enumerate(test_loader):
                if (j+1)%10==0: print("=================================Test Batch"+ str(j+1)+ "===================================================")
                #input_lstm = pad_sequence(sequences=input_lstm,batch_first=True)
                losses_batch,correct_batch= model(input_lstm,input, target, seq_length)
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
        
        with open("/scratch/speech/models/classification/joint_convLSTM_1d.txt","a+") as f:
            #f.write("Epoch: {}-----------Training Loss: {} -------- Testing Loss: {} -------- Training Acc: {} -------- Testing Acc: {}".format(epoch+1,losses,losses_test, accuracy, accuracy_test)+"\n")
            if epoch==epoch_num-1: 
                f.write("Best Accuracy:{:06.5f}".format(max(test_acc))+"\n")
                f.write("Average Top 10 Accuracy:{:06.5f}".format(np.mean(np.sort(np.array(test_acc))[-10:]))+"\n")
                f.write("=============== model ends ==================="+"\n")
    print("success:{}, Best Accuracy:{}".format(path,max(test_acc)))
    '''   
    print("Model Finished:{}".format(path)+"\n")
    print("Best Accuracy:{}".format(max(test_acc))+"\n")
    print("Average Top 10 Accuracy:{}".format(np.mean(np.sort(np.array(test_acc))[-10:]))+"\n")
    '''




'''
    pickle_out=open("/scratch/speech/models/classification/multi_spec_joint_checkpoint_stats_"+path+".pkl","wb")
    pickle.dump({"test_acc":test_acc, "train_acc": train_acc, "train_loss": train_loss,"test_loss":test_loss},pickle_out)
    pickle_out.close()
'''

if __name__ == '__main__':
    args = init_parser()
    train_model(args)