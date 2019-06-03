import config
config.init()
import argparse
import itertools
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import models
from collections import OrderedDict
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from tabulate import tabulate
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from data import *
from metrics import *
from modules.organic import _oConvNd
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
    control_names = [['0'],[config.PARAM['data_name']['train']],['resnet18'],['1']]
    control_names_product = list(itertools.product(*control_names)) 
    model_TAGs = ['_'.join(control_names_product[i]) for i in range(len(control_names_product))]    
    extract_result(model_TAGs)
    gather_result(model_TAGs)
    process_result(control_names)
    show_result()

def extract_result(model_TAGs):
    head = './output/model/milestones_4/02'
    tail = 'best'
    for i in range(len(model_TAGs)):
        model_path = '{}/{}_{}.pkl'.format(head,model_TAGs[i],tail)
        if(os.path.exists(model_path)):
            result = load(model_path)
            save({'train_meter_panel':result['train_meter_panel'],'test_meter_panel':result['test_meter_panel']},'./output/result/{}.pkl'.format(model_TAGs[i])) 
        else:
            print('model path {} not exist'.format(model_path))
    return

def gather_result(model_TAGs):
    gathered_result = {}
    dataset = {}
    dataset['train'],_ = fetch_dataset(data_name=config.PARAM['data_name']['train'])
    config.PARAM['classes_size'] = dataset['train'].classes_size
    data_loader = split_dataset(dataset,data_size=config.PARAM['data_size'],batch_size=config.PARAM['batch_size'])   
    head = './output/result/'
    for i in range(len(model_TAGs)):
        result_path = '{}/{}.pkl'.format(head,model_TAGs[i])
        model_name = model_TAGs[i].split('_')[2]
        if(os.path.exists(result_path)):
            model = eval('models.{}(\'{}\').to(device)'.format(model_name,model_TAGs[i]))
            result = load(result_path)
            summary = summarize(data_loader['train'],model)
            gathered_result[model_TAGs[i]] = {'total_num_params':summary['total_num_params'],'loss':result['test_meter_panel'].panel['loss'].history_avg[-1],
            'acc':result['test_meter_panel'].panel['acc'].history_avg[-1]}
        else:
            print('result path {} not exist'.format(result_path))
    print(gathered_result)
    save(gathered_result,'./output/result/gathered_result.pkl')
    return

def process_result(control_names):
    control_size = [len(control_names[i]) for i in range(len(control_names))]
    result_path = './output/result/gathered_result.pkl'
    result = load(result_path)
    evaluation_names = ['total_num_params','loss','acc']
    processed_result = {}
    processed_result['indices'] = {}
    processed_result['all'] = {k:torch.zeros(control_size,device=config.PARAM['device']) for k in evaluation_names}
    processed_result['mean'] = {}
    processed_result['stderr'] = {}
    for model_TAG in result:
        list_model_TAG = model_TAG.split('_')
        processed_result['indices'][model_TAG] = []
        for i in range(len(control_names)):
            processed_result['indices'][model_TAG].append(control_names[i].index(list_model_TAG[i]))
        print(model_TAG,processed_result['indices'][model_TAG])
        for k in processed_result['all']:
            processed_result['all'][k][tuple(processed_result['indices'][model_TAG])] = result[model_TAG][k]
    for k in evaluation_names:
        processed_result['mean'][k] = processed_result['all'][k].mean(dim=0)
    for k in evaluation_names:
        processed_result['stderr'][k] = processed_result['all'][k].std(dim=0)/math.sqrt(processed_result['all'][k].size(0))
    print(processed_result)
    save(processed_result,'./output/result/processed_result.pkl')
    return

def show_result():
    fig_format = 'png'
    result_path = './output/result/processed_result.pkl'
    result = load(result_path)
    y_name = 'acc'
    num_stderr = 1
    colormap = plt.get_cmap('rainbow')
    band = False
    save = False
    labels = {}
    colors = {}
    colormap_indices = np.linspace(0.2,1,len(result['indices'])).tolist()
    i = 0
    for model_TAG in result['indices']:
        labels[model_TAG] = '{}'.format('_'.join(model_TAG.split('_')[1:]))
        colors[model_TAG] = colormap(colormap_indices[i])
        i += 1
    fig = plt.figure()
    for model_TAG in result['indices']:
        x = result['mean']['total_num_params'][tuple(result['indices'][model_TAG][1:])].cpu().numpy()
        y = result['mean'][y_name][tuple(result['indices'][model_TAG][1:])].cpu().numpy()
        plt.scatter(x,y,color=colors[model_TAG],linestyle='-',label=labels[model_TAG],linewidth=3)
        plt.annotate(labels[model_TAG],(x,y))
        if(band):
            y_max = y + num_stderr * result['stderr'][y_name][tuple(result['indices'][model_TAG][1:])].cpu().numpy()
            y_min = y - num_stderr * result['stderr'][y_name][tuple(result['indices'][model_TAG][1:])].cpu().numpy()
            plt.fill_between(x,y_max,y_min,color=colors[i],alpha=0.5,linewidth=1)
    plt.xlabel('Number of Parameters')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.show()
    makedir_exist_ok('./output/fig')
    fig.savefig('./output/fig/result.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)       
    plt.close()
    return
   
if __name__ == '__main__':
    main()