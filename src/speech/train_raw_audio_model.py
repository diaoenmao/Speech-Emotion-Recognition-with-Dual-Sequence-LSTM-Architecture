import torch
from torch import optim
from raw_audio_model import RawAudioModel
from process_raw_audio_model import IEMOCAP, my_collate
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb
from torch.nn import DataParallel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RawAudioModel(1, 64, 3, 1, 1, 4, 4, 200, 2, 0, 4, 32)
model.cuda()
model=DataParallel(model,device_ids=[0,1,2,3])
model.train()

# Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=0.3, patience=8, threshold=1e-3)

# Load the training data
training_data = IEMOCAP(train=True)
train_loader = DataLoader(dataset=training_data, batch_size=32, shuffle=True, collate_fn=my_collate, num_workers=0)
testing_data = IEMOCAP(train=False)
test_loader = DataLoader(dataset=testing_data, batch_size=32, shuffle=True, collate_fn=my_collate, num_workers=0)

test_acc=[]
train_acc=[]
test_loss=[]
train_loss=[]

for epoch in range(3):  # again, normally you would NOT do 300 epochs, it is toy data
    print("===================================" + str(epoch) + "==============================================")
    losses = 0
    correct=0
    losses_test = 0
    correct_test = 0
    model.train()
    for j, (input, target, seq_length) in enumerate(train_loader):
        print("================================= Batch"+ str(j)+ "===================================================")
        input = input.unsqueeze(1)
        model.zero_grad()
        out, loss = model(input, target, seq_length=seq_length)
        pdb.set_trace()
        losses += loss.item() * target.shape[0]
        loss.backward()
        optimizer.step()

        index = torch.argmax(out, dim=1)
        target_index = torch.argmax(target, dim=1).to(device)
        correct += sum(index == target_index).item()
        print("loss: ",loss)

    accuracy=correct*1.0/len(training_data)
    losses=losses / len(training_data)
    print("accuracy: ", accuracy)
    print("losses: ", losses)

    #after training
    model.eval()
    for test_case, target, seq_length in test_loader:
        out, loss = model(test_case, target, train=False, seq_length=seq_length)
        index = torch.argmax(out, dim=1)
        target_index = torch.argmax(target, dim=1).to(device)
        losses_test += loss.item() * index.shape[0]
        correct_test += sum(index == target_index).item()
    accuracy_test = correct_test * 1.0 / len(testing_data)
    losses_test = losses_test / len(testing_data)

    # data gathering
    test_acc.append(accuracy_test)
    train_acc.append(accuracy)
    test_loss.append(losses_test)
    train_loss.append(losses)

    print("Training Loss: {} -------- Testing Loss: {} -------- Training Acc: {} -------- Testing Acc: {}".format(losses,losses_test, accuracy, accuracy_test))

    scheduler.step(losses)


#with open(stats_path,"a+") as f:
#    f.write("================================="+"Best Test Accuracy"+str(max(test_acc))+"====================================="+"\n")
#pickle_out=open(pickle_path,"wb")
#pickle.dump({"test_acc":test_acc, "train_acc": train_acc, "test_loss": test_loss, "train_loss": train_loss},pickle_out)
#pickle_out.close()

#torch.save(model.state_dict(), model_path)
