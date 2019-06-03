import config
config.init()
import argparse
import datetime
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import models
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
from data import *
from metrics import *
from utils import *


cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if(config.PARAM[k]!=args[k]):
        exec('config.PARAM[\'{0}\'] = {1}'.format(k,args[k]))

def main():
    seeds = list(range(config.PARAM['init_seed'],config.PARAM['init_seed']+config.PARAM['num_Experiments']))
    for i in range(config.PARAM['num_Experiments']):
        model_TAG = '{}_{}_{}'.format(seeds[i],config.PARAM['data_name']['train'],config.PARAM['model_name']) \
            if(config.PARAM['special_TAG']=='') else '{}_{}_{}_{}'.format(seeds[i],config.PARAM['data_name']['train'],config.PARAM['model_name'],config.PARAM['special_TAG'])
        print('Experiment: {}'.format(model_TAG))
        runExperiment(model_TAG)
    return

def runExperiment(model_TAG):
    model_TAG_list = model_TAG.split('_')
    seed = int(model_TAG_list[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    config.PARAM['randomGen'] = np.random.RandomState(seed)
    dataset = {}
    dataset['train'],_ = fetch_dataset(data_name=config.PARAM['data_name']['train'])
    _,dataset['test'] = fetch_dataset(data_name=config.PARAM['data_name']['test'])
    config.PARAM['classes_size'] = dataset['train'].classes_size
    data_loader = split_dataset(dataset,data_size=config.PARAM['data_size'],batch_size=config.PARAM['batch_size'],radomGen=config.PARAM['randomGen'])
    print(config.PARAM)
    print('Training data size {}, Number of Batches {}'.format(config.PARAM['data_size']['train'],len(data_loader['train'])))
    print('Test data size {}, Number of Batches {}'.format(config.PARAM['data_size']['test'],len(data_loader['test'])))
    
    model = eval('models.{}(\'{}\').to(device)'.format(config.PARAM['model_name'],model_TAG))
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    if(config.PARAM['resume_mode'] == 1):
        last_epoch,model,optimizer,scheduler,train_meter_panel,test_meter_panel = resume(model,optimizer,scheduler,model_TAG)              
    elif(config.PARAM['resume_mode'] == 2):
        last_epoch = 0
        _,model,_,_,_,_ = resume(model,optimizer,scheduler,model_TAG)
        train_meter_panel = Meter_Panel(config.PARAM['metric_names']['train'])
        test_meter_panel = Meter_Panel(config.PARAM['metric_names']['test'])
    else:
        last_epoch = 0
        train_meter_panel = Meter_Panel(config.PARAM['metric_names']['train'])
        test_meter_panel = Meter_Panel(config.PARAM['metric_names']['test'])
    model = nn.DataParallel(model,device_ids=list(range(world_size))) if(world_size > 1) else model
    best_pivot = 65535
    best_pivot_name = 'loss'
    for epoch in range(last_epoch, config.PARAM['num_epochs']+1):
        cur_train_meter_panel = train(data_loader['train'],model,optimizer,epoch,model_TAG)
        cur_test_meter_panel = test(data_loader['test'],model,epoch,model_TAG)
        print_result(model_TAG,epoch,cur_train_meter_panel,cur_test_meter_panel)
        if(config.PARAM['scheduler_name'] == 'ReduceLROnPlateau'):
            scheduler.step(metrics=cur_test_meter_panel.panel['loss'].avg,epoch=epoch+1)
        else:
            scheduler.step(epoch=epoch+1)
        train_meter_panel.update(cur_train_meter_panel)
        test_meter_panel.update(cur_test_meter_panel)
        if(config.PARAM['save_mode']>=0):
            model_state_dict = model.module.state_dict() if(world_size > 1) else model.state_dict()
            save_result = {'config':config.PARAM,'epoch':epoch+1,'model_dict':model_state_dict,'optimizer_dict':optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict(),'train_meter_panel':train_meter_panel,'test_meter_panel':test_meter_panel}
            save(save_result,'./output/model/{}_checkpoint.pkl'.format(model_TAG))
            if(best_pivot > cur_test_meter_panel.panel[best_pivot_name].avg):
                best_pivot = cur_test_meter_panel.panel[best_pivot_name].avg
                save(save_result,'./output/model/{}_best.pkl'.format(model_TAG))
    return

def train(train_loader,model,optimizer,epoch,model_TAG):
    meter_panel = Meter_Panel(config.PARAM['metric_names']['train'])
    model.train(True)
    end = time.time()
    for i, input in enumerate(train_loader):
        input = collate(input)
        input = dict_to_device(input,device)
        output = model(input)
        output['loss'] = torch.mean(output['loss']) if(world_size > 1) else output['loss']                                                                                          
        optimizer.zero_grad()
        output['loss'].backward()
        optimizer.step()
        evaluation = meter_panel.eval(input,output,config.PARAM['metric_names']['train'])
        batch_time = time.time() - end
        meter_panel.update(evaluation,len(input['img']))
        meter_panel.update({'batch_time':batch_time})
        end = time.time()
        if(i % (len(train_loader)//5) == 0):
            estimated_finish_time = str(datetime.timedelta(seconds=(len(train_loader)-i-1)*batch_time))
            print('Train Epoch({}): {}[({:.0f}%)]{}, Estimated Finish Time: {}'.format(
                model_TAG, epoch, 100. * i / len(train_loader), meter_panel.summary(['loss','batch_time'] + config.PARAM['metric_names']['train']), estimated_finish_time))
    return meter_panel
            
def test(validation_loader,model,epoch,model_TAG):
    meter_panel = Meter_Panel(config.PARAM['metric_names']['test'])
    with torch.no_grad():
        model.train(False)
        end = time.time()
        for i, input in enumerate(validation_loader):
            input = collate(input)
            input = dict_to_device(input,device)
            output = model(input)
            output['loss'] = torch.mean(output['loss']) if(world_size > 1) else output['loss']
            evaluation = meter_panel.eval(input,output,config.PARAM['metric_names']['test'])
            batch_time = time.time() - end
            meter_panel.update(evaluation,len(input['img']))
            meter_panel.update({'batch_time':batch_time})
            end = time.time()
    return meter_panel

def make_optimizer(model):
    if(config.PARAM['optimizer_name']=='Adam'):
        optimizer = optim.Adam(model.parameters(),lr=config.PARAM['lr'])
    elif(config.PARAM['optimizer_name']=='SGD'):
        optimizer = optim.SGD(model.parameters(),lr=config.PARAM['lr'], momentum=0.9, weight_decay=config.PARAM['weight_decay'])
    else:
        raise ValueError('Optimizer name not supported')
    return optimizer
    
def make_scheduler(optimizer):
    if(config.PARAM['scheduler_name']=='MultiStepLR'):
        scheduler = MultiStepLR(optimizer,milestones=config.PARAM['milestones'],gamma=config.PARAM['factor'])
    elif(config.PARAM['scheduler_name']=='ReduceLROnPlateau'):
        scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=config.PARAM['factor'],verbose=True,threshold=config.PARAM['threshold'],threshold_mode=config.PARAM['threshold_mode'])
    elif(config.PARAM['scheduler_name']=='CosineAnnealingLR'):
        scheduler = CosineAnnealingLR(optimizer, config.PARAM['num_epochs'])
    else:
        raise ValueError('Scheduler_name name not supported')
    return scheduler

def collate(input):
    for k in input:
        input[k] = torch.stack(input[k],0)
    return input
 
def print_result(model_TAG,epoch,train_meter_panel,test_meter_panel):
    estimated_finish_time = str(datetime.timedelta(seconds=(config.PARAM['num_epochs'] - epoch)*train_meter_panel.panel['batch_time'].sum))
    print('***Test Epoch({}): {}{}{}, Estimated Finish Time: {}'.format(model_TAG,epoch,test_meter_panel.summary(['loss']+config.PARAM['metric_names']['test']),train_meter_panel.summary(['batch_time']),estimated_finish_time))
    return

def resume(model,optimizer,scheduler,model_TAG):
    if(os.path.exists('./output/model/{}_checkpoint.pkl'.format(model_TAG))):
        checkpoint = load('./output/model/{}_checkpoint.pkl'.format(model_TAG))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        train_meter_panel = checkpoint['train_meter_panel']
        test_meter_panel = checkpoint['test_meter_panel']
        print('Resume from {}'.format(last_epoch))
    else:
        last_epoch = 0
        print('Not found existing model, and start from epoch {}'.format(last_epoch))
    return last_epoch,model,optimizer,scheduler,train_meter_panel,test_meter_panel
    
if __name__ == "__main__":
    main()