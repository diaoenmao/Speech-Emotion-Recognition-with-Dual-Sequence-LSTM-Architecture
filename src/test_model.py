import config
config.init()
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import models
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
        result = runExperiment(model_TAG)
        save({'test_meter_panel':result},'./output/result/{}.pkl'.format(model_TAG)) 
    return

def runExperiment(model_TAG):
    model_TAG_list = model_TAG.split('_')
    seed = int(model_TAG_list[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    randomGen = np.random.RandomState(seed)

    dataset = {}
    _,dataset['test'] = fetch_dataset(data_name=config.PARAM['data_name']['test'])
    config.PARAM['classes_size'] = dataset['test'].classes_size
    data_loader = split_dataset(dataset,data_size=config.PARAM['data_size'],batch_size=config.PARAM['batch_size'],radomGen=randomGen)
    print(config.PARAM)    
    print('Test data size {}, Number of Batches {}'.format(config.PARAM['data_size']['test'],len(data_loader['test']))) 
    
    model = eval('models.{}(\'{}\').to(device)'.format(config.PARAM['model_name'],model_TAG))
    best = load('./output/model/{}_best.pkl'.format(model_TAG))
    last_epoch = best['epoch']
    print('Test from {}'.format(last_epoch))
    model.load_state_dict(best['model_dict'])
    result = test(data_loader['test'],model,last_epoch,model_TAG)
    print_result(model_TAG,last_epoch,result) 
    return result
            
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
            meter_panel.update(evaluation,batch_size)
            meter_panel.update({'batch_time':batch_time})
            end = time.time()
    return meter_panel
     
def collate(input):
    for k in input:
        input[k] = torch.stack(input[k],0)
    return input
        
def print_result(model_TAG,epoch,result):
    print('Test Epoch({}): {}({}_{}){}'.format(model_TAG,epoch,result.summary(['loss']+config.PARAM['metric_names']['test'])))
    return
    
if __name__ == "__main__":
    main()    