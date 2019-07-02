import torch
from torch import optim
from spectrogram_model import SpectrogramModel
from process_spectrogram_model import IEMOCAP, my_collate
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR as Cos
import pdb
from torch.nn import DataParallel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SpectrogramModel(3, 64, 3, 1, 1, 4, 4, 200, 2, 0.2, 4, 100, True)
model.cuda()
model=DataParallel(model,device_ids=[0,1,2,3])
model.train()

# Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
optimizer = optim.SGD(model.parameters(), lr=0.1)
#scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=0.3, patience=5, threshold=1e-3)
scheduler =Cos(optimizer, T_max=100, eta_min=0.0001)
# Load the training data
training_data = IEMOCAP(train=True)
train_loader = DataLoader(dataset=training_data, batch_size=100, shuffle=True, collate_fn=my_collate, num_workers=0)
testing_data = IEMOCAP(train=False)
test_loader = DataLoader(dataset=testing_data, batch_size=100, shuffle=True, collate_fn=my_collate, num_workers=0)

test_acc=[]
train_acc=[]
test_loss=[]
train_loss=[]

for epoch in range(200):  # again, normally you would NOT do 300 epochs, it is toy data
    print("===================================" + str(epoch+1) + "==============================================")
    losses = 0
    correct=0
    losses_test = 0
    correct_test = 0
    model.train()
    for j, (input, target) in enumerate(train_loader):
       # if (j+1)%50==0: print("================================= Batch"+ str(j+1)+ "===================================================")
        #input=input.float()
        #input = input.unsqueeze(1)
        pdb.set_trace()
        input = input.permute(0, 3, 1, 2)
        model.zero_grad()
        out, loss = model(input, target)
        loss = torch.mean(loss)
        losses += loss.item() * target.shape[0]
        loss.backward()
        optimizer.step()

        index = torch.argmax(out, dim=1)
        target_index = torch.argmax(target, dim=1).to(device)
        correct += sum(index == target_index).item()

    accuracy=correct*1.0/len(training_data)
    losses=losses / len(training_data)

    model.eval()
    for test_case, target, seq_length in test_loader:
        test_case=test_case.float()
        #test_case = test_case.unsqueeze(1)
        out, loss = model(test_case, target, train=False)
        index = torch.argmax(out, dim=1)
        target_index = torch.argmax(target, dim=1).to(device)
        loss = torch.mean(loss)
        losses_test += loss.item() * index.shape[0]
        correct_test += sum(index == target_index).item()
    accuracy_test = correct_test * 1.0 / len(testing_data)
    losses_test = losses_test / len(testing_data)

    # data gathering
    test_acc.append(accuracy_test)
    train_acc.append(accuracy)
    test_loss.append(losses_test)
    train_loss.append(losses)
    print("Epoch: {}-----------Training Loss: {} -------- Testing Loss: {} -------- Training Acc: {} -------- Testing Acc: {}".format(epoch+1,losses,losses_test, accuracy, accuracy_test)+"\n")
    #with open("/scratch/speech/models/classification/spectrogram_stats.txt","a+") as f:
        #f.write("Epoch: {}-----------Training Loss: {} -------- Testing Loss: {} -------- Training Acc: {} -------- Testing Acc: {}".format(epoch+1,losses,losses_test, accuracy, accuracy_test)+"\n")


    scheduler.step()


#pickle_out=open("/scratch/speech/models/classification/spectrogram_checkpoint_stats.pkl","wb")
#pickle.dump({"test_acc":test_acc, "train_acc": train_acc, "test_loss": test_loss, "train_loss": train_loss},pickle_out)
#pickle_out.close()

#torch.save(model.state_dict(), model_path)
