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

def func1():
	model = ConvLSTM(1, [64,32,16],[9,5,5],100)
	model.cuda()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	epoch=1
	#pdb.set_trace()
	#model.load_state_dict(torch.load("/scratch/speech/models/classification/ConvLSTM_checkpoint_epoch_{}.pt".format(epoch))['state_dict'])
	path="/scratch/speech/models/classification/ConvLSTM_checkpoint_epoch_{}.pt".format(epoch)


	state_dict = torch.load(path)
	# create new OrderedDict that does not contain `module.`
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
	    name = k[7:] # remove `module.`
	    new_state_dict[name] = v
	# load params
	model.load_state_dict(new_state_dict)
	#model=DataParallel(model,device_ids=[0,1,2,3])
	training_data = IEMOCAP(train=True)
	train_loader = DataLoader(dataset=training_data, batch_size=15, shuffle=True, collate_fn=my_collate, num_workers=0)
	testing_data = IEMOCAP(train=False)
	test_loader = DataLoader(dataset=testing_data, batch_size=15, shuffle=True, collate_fn=my_collate, num_workers=0)

	test_acc=[]
	train_acc=[]
	test_loss=[]
	train_loss=[]
	losses_test=0
	correct_test=0
	loss=0
	model.eval()
	for j,(test_case, target, seq_length) in enumerate(test_loader):
	    print(j)
	    test_case=test_case.float()
	    test_case = test_case.unsqueeze(1)
	    test_case=torch.split(test_case,1280,dim=2)
	    try:
	        out, loss = model(test_case, target)
	    except:
	    	pdb.set_trace()

	    loss = torch.mean(loss,dim=0)
	    out=torch.flatten(out,start_dim=0,end_dim=1)

	    index = torch.argmax(out, dim=1)
	    target_index = torch.argmax(target, dim=1).to(device)
	    loss = torch.mean(loss)
	    losses_test += loss.item() * index.shape[0]
	    correct_test += sum(index == target_index).item()
	accuracy_test = correct_test * 1.0 / len(testing_data)
	losses_test = losses_test / len(testing_data)
	print(accuracy_test)
	print(losses_test)
def func2():
	