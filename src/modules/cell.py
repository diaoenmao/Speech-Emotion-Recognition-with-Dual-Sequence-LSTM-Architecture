import config
import copy
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict 
from modules.shuffle import PixelUnShuffle,PixelShuffle
from modules.organic import oConv2d
from utils import _ntuple,gumbel_softmax,gumbel_softrank

device = config.PARAM['device']

def Normalization(cell_info):
    if(cell_info['mode']=='none'):
        return nn.Sequential()
    elif(cell_info['mode']=='bn'):
        return nn.BatchNorm2d(cell_info['input_size'])
    elif(cell_info['mode']=='in'):
        return nn.InstanceNorm2d(cell_info['input_size'])
    else:
        raise ValueError('Normalization mode not supported')
    return
    
def Activation(cell_info):
    if(cell_info['mode']=='none'):
        return nn.Sequential()
    elif(cell_info['mode']=='tanh'):
        return nn.Tanh()
    elif(cell_info['mode']=='relu'):
        return nn.ReLU(inplace=True)
    elif(cell_info['mode']=='prelu'):
        return nn.PReLU()
    elif(cell_info['mode']=='elu'):
        return nn.ELU(inplace=True)
    elif(cell_info['mode']=='selu'):
        return nn.SELU(inplace=True)
    elif(cell_info['mode']=='celu'):
        return nn.CELU(inplace=True)
    elif(cell_info['mode']=='sigmoid'):
        return nn.Sigmoid()
    elif(cell_info['mode']=='softmax'):
        return nn.SoftMax()
    else:
        raise ValueError('Activation mode not supported')
    return

class BasicCell(nn.Module):
    def __init__(self, cell_info):
        super(BasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        if(cell_info['mode']=='down'):
            cell_in_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                        'kernel_size':3,'stride':2,'padding':1,'dilation':1,'groups':cell_info['groups'],'bias':cell_info['bias']}
        elif(cell_info['mode']=='downsample'):
            cell_in_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                        'kernel_size':2,'stride':2,'padding':0,'dilation':1,'groups':cell_info['groups'],'bias':cell_info['bias']}
        elif(cell_info['mode']=='pass'):
            cell_in_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                        'kernel_size':3,'stride':1,'padding':1,'dilation':1,'groups':cell_info['groups'],'bias':cell_info['bias']}
        elif(cell_info['mode']=='upsample'):
            cell_in_info = {'cell':'ConvTranspose2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                        'kernel_size':2,'stride':2,'padding':0,'output_padding':0,'dilation':1,'groups':cell_info['groups'],'bias':cell_info['bias']}
        elif(cell_info['mode']=='fc'):
            cell_in_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                        'kernel_size':1,'stride':1,'padding':0,'dilation':1,'groups':cell_info['groups'],'bias':cell_info['bias']}
        elif(cell_info['mode']=='fc_down'):
            cell_in_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                        'kernel_size':1,'stride':2,'padding':0,'dilation':1,'groups':cell_info['groups'],'bias':cell_info['bias']}
        else:
            raise ValueError('model mode not supported')
        cell['in'] = Cell(cell_in_info)
        cell['activation'] = Cell({'cell':'Activation','mode':cell_info['activation']})
        cell['normalization'] = Cell({'cell':'Normalization','input_size':cell_info['input_size'],'mode':cell_info['normalization']}) if(cell_info['order']=='before') else \
        Cell({'cell':'Normalization','input_size':cell_info['output_size'],'mode':cell_info['normalization']})
        return cell
        
    def forward(self, input):
        x = input
        if(self.cell_info['order']=='before'):
            x = self.cell['in'](self.cell['activation'](self.cell['normalization'](x)))
        elif(self.cell_info['order']=='after'):
            x = self.cell['activation'](self.cell['normalization'](self.cell['in'](x)))
        else:
            raise ValueError('wrong order')
        return x

class ResBasicCell(nn.Module):
    def __init__(self, cell_info):
        super(ResBasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(cell_info['mode'] == 'down'):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc_down',
                'normalization':cell_info['normalization'],'activation':'none'}
            elif(cell_info['input_size'] != cell_info['output_size']):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc',
                'normalization':cell_info['normalization'],'activation':'none'}
            else:
                cell_shortcut_info = {'cell':'none'}
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':cell_info['mode'],
            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            cell_out_info = {'input_size':cell_info['output_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'pass',
            'normalization':cell_info['normalization'],'activation':'none'}
            cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell[i]['activation'] = Cell({'cell':'Activation','mode':cell_info['activation']})
            cell_info['input_size'] = cell_info['output_size']
            cell_info['mode'] = 'pass'
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = self.cell[i]['shortcut'](x)
            x = self.cell[i]['in'](x)
            x = self.cell[i]['out'](x)
            x = self.cell[i]['activation'](x+shortcut)
        return x

class GroupResBasicCell(nn.Module):
    def __init__(self, cell_info):
        super(GroupResBasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(cell_info['mode'] == 'down'):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc_down',
                'normalization':cell_info['normalization'],'activation':'none'}
            elif(cell_info['input_size'] != cell_info['output_size']):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc',
                'normalization':cell_info['normalization'],'activation':'none'}
            else:
                cell_shortcut_info = {'cell':'none'}
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':cell_info['mode'],
            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            cell_out_info = {'input_size':cell_info['output_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'pass',
            'normalization':cell_info['normalization'],'activation':'none','groups':cell_info['groups']}
            cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell[i]['activation'] = cell[i]['activation'] = Cell({'cell':'Activation','mode':cell_info['activation']})
            cell_info['input_size'] = cell_info['output_size']
            cell_info['mode'] = 'pass'
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = self.cell[i]['shortcut'](x)
            x = self.cell[i]['in'](x)
            x = self.cell[i]['out'](x)
            x = self.cell[i]['activation'](x+shortcut)
        return x
        
class ShuffleGroupResBasicCell(nn.Module):
    def __init__(self, cell_info):
        super(ShuffleGroupResBasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(cell_info['mode'] == 'down'):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc_down',
                'normalization':cell_info['normalization'],'activation':'none'}
            elif(cell_info['input_size'] != cell_info['output_size']):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc',
                'normalization':cell_info['normalization'],'activation':'none'}
            else:
                cell_shortcut_info = {'cell':'none'}
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':cell_info['mode'],
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'groups':cell_info['input_size']//cell_info['groups']}
            cell_out_info = {'input_size':cell_info['output_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'pass',
            'normalization':cell_info['normalization'],'activation':'none','groups':cell_info['groups']}
            cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell[i]['activation'] = cell[i]['activation'] = Cell({'cell':'Activation','mode':cell_info['activation']})
            cell[i]['in_shuffle'] = ShuffleCell({'input_size':[-1,cell_info['groups']],'dim':1,'permutation':[1,0]})
            cell[i]['out_shuffle'] = ShuffleCell({'input_size':[cell_info['groups'],-1],'dim':1,'permutation':[1,0]})
            cell_info['input_size'] = cell_info['output_size']
            cell_info['mode'] = 'pass'
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = self.cell[i]['shortcut'](x)
            x = self.cell[i]['in'](x)
            x = self.cell[i]['in_shuffle'](x)
            x = self.cell[i]['out'](x)
            x = self.cell[i]['out_shuffle'](x)            
            x = self.cell[i]['activation'](x+shortcut)
        return x
        
class BottleNeckCell(nn.Module):
    def __init__(self, cell_info):
        super(BottleNeckCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(cell_info['mode'] == 'down'):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc_down',
                'normalization':cell_info['normalization'],'activation':'none'}
            elif(cell_info['input_size'] != cell_info['output_size']):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc',
                'normalization':cell_info['normalization'],'activation':'none'}
            else:
                cell_shortcut_info = {'cell':'none'} 
            cell_reduce_info = {'input_size':cell_info['input_size'],'output_size':cell_info['neck_in_size'],'cell':'BasicCell','mode':'fc',
            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            cell_neck_info = {'input_size':cell_info['neck_in_size'],'output_size':cell_info['neck_out_size'],'cell':'BasicCell','mode':cell_info['mode'],
            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            cell_expand_info = {'input_size':cell_info['neck_out_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc',
            'normalization':cell_info['normalization'],'activation':'none'}
            cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell[i]['reduce'] = Cell(cell_reduce_info)
            cell[i]['neck'] = Cell(cell_neck_info)
            cell[i]['expand'] = Cell(cell_expand_info)
            cell[i]['activation'] = Cell({'cell':'Activation','mode':cell_info['activation']})
            cell_info['input_size'] = cell_info['output_size']
            cell_info['mode'] = 'pass'
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = self.cell[i]['shortcut'](x)
            x = self.cell[i]['reduce'](x)
            x = self.cell[i]['neck'](x)
            x = self.cell[i]['expand'](x)
            x = self.cell[i]['activation'](x+shortcut)
        return x

class GroupBottleNeckCell(nn.Module):
    def __init__(self, cell_info):
        super(GroupBottleNeckCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(cell_info['mode'] == 'down'):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc_down',
                'normalization':cell_info['normalization'],'activation':'none'}
            elif(cell_info['input_size'] != cell_info['output_size']):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc',
                'normalization':cell_info['normalization'],'activation':'none'}
            else:
                cell_shortcut_info = {'cell':'none'} 
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['neck_in_size'],'cell':'BasicCell','mode':'fc',
            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            cell_neck_info = {'input_size':cell_info['neck_in_size'],'output_size':cell_info['neck_out_size'],'cell':'BasicCell','mode':cell_info['mode'],
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'groups':cell_info['groups']}
            cell_out_info = {'input_size':cell_info['neck_out_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc',
            'normalization':cell_info['normalization'],'activation':'none'}
            cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['neck'] = Cell(cell_neck_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell[i]['activation'] = Cell({'cell':'Activation','mode':cell_info['activation']})
            cell_info['input_size'] = cell_info['output_size']
            cell_info['mode'] = 'pass'
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = self.cell[i]['shortcut'](x)
            x = self.cell[i]['in'](x)
            x = self.cell[i]['neck'](x)
            x = self.cell[i]['out'](x)
            x = self.cell[i]['activation'](x+shortcut)
        return x

class ShuffleGroupBottleNeckCell(nn.Module):
    def __init__(self, cell_info):
        super(ShuffleGroupBottleNeckCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(cell_info['mode'] == 'down'):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc_down',
                'normalization':cell_info['normalization'],'activation':'none'}
            elif(cell_info['input_size'] != cell_info['output_size']):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc',
                'normalization':cell_info['normalization'],'activation':'none'}
            else:
                cell_shortcut_info = {'cell':'none'} 
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['neck_in_size'],'cell':'BasicCell','mode':'fc',
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'groups':cell_info['neck_in_size']//cell_info['groups']}
            cell_neck_info = {'input_size':cell_info['neck_in_size'],'output_size':cell_info['neck_out_size'],'cell':'BasicCell','mode':cell_info['mode'],
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'groups':cell_info['groups']}
            cell_out_info = {'input_size':cell_info['neck_out_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'fc',
            'normalization':cell_info['normalization'],'activation':'none'}
            cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['neck'] = Cell(cell_neck_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell[i]['activation'] = Cell({'cell':'Activation','mode':cell_info['activation']})
            cell[i]['in_shuffle'] = ShuffleCell({'input_size':[-1,cell_info['groups']],'dim':1,'permutation':[1,0]})
            cell[i]['out_shuffle'] = ShuffleCell({'input_size':[cell_info['groups'],-1],'dim':1,'permutation':[1,0]})
            cell_info['input_size'] = cell_info['output_size']
            cell_info['mode'] = 'pass'
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = self.cell[i]['shortcut'](x)
            x = self.cell[i]['in'](x)
            x = self.cell[i]['in_shuffle'](x)
            x = self.cell[i]['neck'](x)
            x = self.cell[i]['out'](x)
            x = self.cell[i]['out_shuffle'](x)
            x = self.cell[i]['activation'](x+shortcut)
        return x

class DenseCell(nn.Module):
    def __init__(self, cell_info):
        super(DenseCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['bottleneck']*cell_info['growth_rate'],'cell':'BasicCell','mode':'fc',
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'order':'before'}
            cell_out_info = {'input_size':cell_info['bottleneck']*cell_info['growth_rate'],'output_size':cell_info['growth_rate'],'cell':'BasicCell','mode':'pass',
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'order':'before'}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell_info['input_size'] = cell_info['input_size'] + cell_info['growth_rate']
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = x
            x = self.cell[i]['in'](x)
            x = self.cell[i]['out'](x)
            x = torch.cat([shortcut,x], dim=1)
        return x

class GroupDenseCell(nn.Module):
    def __init__(self, cell_info):
        super(GroupDenseCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['bottleneck']*cell_info['growth_rate'],'cell':'BasicCell','mode':'fc',
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'order':'before'}
            cell_out_info = {'input_size':cell_info['bottleneck']*cell_info['growth_rate'],'output_size':cell_info['growth_rate'],'cell':'BasicCell','mode':'pass',
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'groups':cell_info['groups'],'order':'before'}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell_info['input_size'] = cell_info['input_size'] + cell_info['growth_rate']
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = x
            x = self.cell[i]['in'](x)
            x = self.cell[i]['out'](x)
            x = torch.cat([shortcut,x], dim=1)
        return x

class ShuffleGroupDenseCell(nn.Module):
    def __init__(self, cell_info):
        super(ShuffleGroupDenseCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['bottleneck']*cell_info['growth_rate'],'cell':'BasicCell','mode':'fc',
            'normalization':'none','activation':'none','groups':cell_info['bottleneck']*cell_info['growth_rate']//cell_info['groups'],'order':'before'}
            cell_out_info = {'input_size':cell_info['bottleneck']*cell_info['growth_rate'],'output_size':cell_info['growth_rate'],'cell':'BasicCell','mode':'pass',
            'normalization':'none','activation':'none','groups':cell_info['groups'],'order':'before'}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell[i]['in_shuffle'] = ShuffleCell({'input_size':[-1,cell_info['groups']],'dim':1,'permutation':[1,0]})
            cell[i]['out_shuffle'] = ShuffleCell({'input_size':[cell_info['groups'],-1],'dim':1,'permutation':[1,0]})
            cell_info['input_size'] = cell_info['input_size'] + cell_info['growth_rate']
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = x
            x = self.cell[i]['in'](x)
            x = self.cell[i]['in_shuffle'](x)                                    
            x = self.cell[i]['out'](x)
            x = self.cell[i]['out_shuffle'](x)
            x = torch.cat([shortcut,x], dim=1)
        return x

class LSTMCell(nn.Module):
    def __init__(self, cell_info):
        super(LSTMCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        _tuple = _ntuple(2)
        cell_info['activation'] = _tuple(cell_info['activation'])
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            cell_in_info = {**cell_info['in'][i],'output_size':4*cell_info['in'][i]['output_size']}
            cell_hidden_info = {**cell_info['hidden'][i],'output_size':4*cell_info['hidden'][i]['output_size']}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['hidden'] = Cell(cell_hidden_info)
            cell[i]['activation'] = nn.ModuleList([Activation(cell_info['activation'][0]),Activation(cell_info['activation'][1])])
        return cell
        
    def init_hidden(self, hidden_size):
        hidden = [[torch.zeros(hidden_size,device=device)],[torch.zeros(hidden_size,device=device)]]
        return hidden
    
    def free_hidden(self):
        self.hidden = None
        return
        
    def forward(self, input, hidden=None):
        x = input
        x = x.unsqueeze(1) if(input.dim()==4) else x
        hx,cx = [None for _ in range(len(self.cell))],[None for _ in range(len(self.cell))]
        for i in range(len(self.cell)):
            y = [None for _ in range(x.size(1))]
            for j in range(x.size(1)):
                gates = self.cell[i]['in'](x[:,j])
                if(hidden is None):
                    if(self.hidden is None):
                        self.hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                    else:
                        if(i==len(self.hidden[0])):
                            tmp_hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                            self.hidden[0].extend(tmp_hidden[0])
                            self.hidden[1].extend(tmp_hidden[1])
                        else:
                            pass
                if(j==0):
                    hx[i],cx[i] = self.hidden[0][i],self.hidden[1][i]
                gates += self.cell[i]['hidden'](hx[i])
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = self.cell[i]['activation'][0](cellgate)
                outgate = torch.sigmoid(outgate)
                cx[i] = (forgetgate * cx[i]) + (ingate * cellgate)
                hx[i] = outgate * self.cell[i]['activation'][1](cx[i])                
                y[j] = hx[i]
            x = torch.stack(y,dim=1)
        self.hidden = [hx,cx]
        x = x.squeeze(1) if(input.dim()==4) else x
        return x

class ResLSTMCell(nn.Module):
    def __init__(self, cell_info):
        super(ResLSTMCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        _tuple = _ntuple(2)
        cell_info['activation'] = _tuple(cell_info['activation'])
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(i==0):
                cell_shortcut_info = cell_info['shortcut'][i]
                cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell_in_info = {**cell_info['in'][i],'output_size':4*cell_info['in'][i]['output_size']}
            cell_hidden_info = {**cell_info['hidden'][i],'output_size':4*cell_info['hidden'][i]['output_size']}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['hidden'] = Cell(cell_hidden_info)            
            cell[i]['activation'] = nn.ModuleList([Activation(cell_info['activation'][0]),Activation(cell_info['activation'][1])])         
        return cell
        
    def init_hidden(self, hidden_size):
        hidden = [[torch.zeros(hidden_size,device=device)],[torch.zeros(hidden_size,device=device)]]
        return hidden
    
    def free_hidden(self):
        self.hidden = None
        return

    def forward(self, input, hidden=None):
        x = input
        x = x.unsqueeze(1) if(input.dim()==4) else x
        hx,cx = [None for _ in range(len(self.cell))],[None for _ in range(len(self.cell))]
        shortcut = [None for _ in range(x.size(1))]
        for i in range(len(self.cell)):
            y = [None for _ in range(x.size(1))]
            for j in range(x.size(1)):
                if(i==0):
                    shortcut[j] = self.cell[i]['shortcut'](x[:,j])
                gates = self.cell[i]['in'](x[:,j])
                if(hidden is None):
                    if(self.hidden is None):
                        self.hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                    else:
                        if(i==len(self.hidden[0])):
                            tmp_hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                            self.hidden[0].extend(tmp_hidden[0])
                            self.hidden[1].extend(tmp_hidden[1])
                        else:
                            pass
                else:
                    self.hidden = hidden
                if(j==0):
                    hx[i],cx[i] = self.hidden[0][i],self.hidden[1][i]
                gates += self.cell[i]['hidden'](hx[i])
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = self.cell[i]['activation'][0](cellgate)
                outgate = torch.sigmoid(outgate)
                cx[i] = (forgetgate * cx[i]) + (ingate * cellgate)  
                hx[i] = outgate * self.cell[i]['activation'][1](cx[i]) if(i<len(self.cell)-1) else outgate*(shortcut[j] + self.cell[i]['activation'][1](cx[i]))
                y[j] = hx[i]
            x = torch.stack(y,dim=1)
        self.hidden = [hx,cx]
        x = x.squeeze(1) if(input.dim()==4) else x
        return x

class ShuffleCell(nn.Module):
    def __init__(self, cell_info):
        super(ShuffleCell, self).__init__()
        self.cell_info = cell_info
        
    def forward(self, input):
        input_size = [*input.size()[:self.cell_info['dim']],*self.cell_info['input_size'],*input.size()[(self.cell_info['dim']+1):]]
        permutation = [i for i in range(len(input.size()[:self.cell_info['dim']]))] + \
                    [self.cell_info['permutation'][i]+self.cell_info['dim'] for i in range(len(self.cell_info['permutation']))] + \
                    [i+self.cell_info['dim']+len(self.cell_info['input_size']) for i in range(len(input.size()[(self.cell_info['dim']+1):]))]
        output_size = [*input.size()[:self.cell_info['dim']],-1,*input.size()[(self.cell_info['dim']+1):]]
        x = input.reshape(input_size)
        x = x.permute(permutation)
        x = x.reshape(output_size)
        return x

class PixelShuffleCell(nn.Module):
    def __init__(self, cell_info):
        super(PixelShuffleCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        if(cell_info['mode'] == 'down'):        
            cell = PixelUnShuffle(cell_info['scale_factor'])
        elif(cell_info['mode'] == 'up'):        
            cell = PixelShuffle(cell_info['scale_factor'])
        else:
            raise ValueError('model mode not supported')
        return cell
        
    def forward(self, input):
        x = self.cell(input)
        return x

class PoolCell(nn.Module):
    def __init__(self, cell_info):
        super(PoolCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        if(cell_info['mode'] == 'avg'):
            cell = nn.AvgPool2d(kernel_size=cell_info['kernel_size'],stride=cell_info['stride'],padding=cell_info['padding'],
            ceil_mode=cell_info['ceil_mode'],count_include_pad=cell_info['count_include_pad'])
        elif(cell_info['mode'] == 'max'):
            cell = nn.MaxPool2d(kernel_size=cell_info['kernel_size'],stride=cell_info['stride'],padding=cell_info['padding'],
            dilation=cell_info['dilation'],return_indices=cell_info['return_indices'],ceil_mode=cell_info['ceil_mode'])
        elif(cell_info['mode'] == 'maxun'):
            cell = nn.MaxUnPool2d(kernel_size=cell_info['kernel_size'],stride=cell_info['stride'],padding=cell_info['padding'])
        elif(cell_info['mode'] == 'adapt_avg'):
            cell = nn.AdaptiveAvgPool2d(cell_info['output_size'])
        elif(cell_info['mode'] == 'adapt_max'):
            cell = nn.AdaptiveMaxPool2d(cell_info['output_size'])
        else:
            raise ValueError('model mode not supported')
        return cell
        
    def forward(self, input):
        x = self.cell(input)
        return x

class DownTransitionCell(nn.Module):
    def __init__(self, cell_info):
        super(DownTransitionCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = []
        if(cell_info['mode'] == 'cnn'):
            cell.append(Cell({'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'downsample',
                'normalization':cell_info['normalization'],'activation':cell_info['activation'],'order':cell_info['order']}))
        elif(cell_info['mode'] == 'avg'):
            cell.append(Cell({'cell':'PoolCell','mode':'avg','kernel_size':2}))
        elif(cell_info['mode'] == 'max'):
            cell.append(Cell({'cell':'PoolCell','mode':'max','kernel_size':2}))
        elif(cell_info['mode'] == 'pixelshuffle'):
            cell.append(Cell({'cell':'PixelShuffleCell','mode':'down','scale_factor':2}))
        else:
            raise ValueError('model mode not supported')
        cell = nn.Sequential(*cell)
        return cell
        
    def forward(self, input):
        x = input
        x = self.cell(x)
        return x

class UpTransitionCell(nn.Module):
    def __init__(self, cell_info):
        super(UpTransitionCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = []
        if(cell_info['mode'] == 'cnn'):
            cell.append(Cell({'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'BasicCell','mode':'upsample',
                'normalization':cell_info['normalization'],'activation':cell_info['activation'],'order':cell_info['order']}))
        elif(cell_info['mode'] == 'max'):
            cell.append(Cell({'cell':'PoolCell','mode':'maxun','kernel_size':2}))
        elif(cell_info['mode'] == 'pixelshuffle'):
            cell.append(Cell({'cell':'PixelShuffleCell','mode':'up','scale_factor':2}))
        else:
            raise ValueError('model mode not supported')
        cell = nn.Sequential(*cell)
        return cell
        
    def forward(self, input):
        x = input
        x = self.cell(x)
        return x

class CartesianBasicCell(nn.Module):
    def __init__(self, cell_info):
        super(CartesianBasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        if(cell_info['mode']=='down'):
            cell_in_info = {'cell':'oConv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size']*cell_info['cardinality'],
                        'kernel_size':3,'stride':2,'padding':1,'dilation':1,'groups':1,'bias':cell_info['bias']}
        elif(cell_info['mode']=='downsample'):
            cell_in_info = {'cell':'oConv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size']*cell_info['cardinality'],
                        'kernel_size':2,'stride':2,'padding':0,'dilation':1,'groups':1,'bias':cell_info['bias']}
        elif(cell_info['mode']=='pass'):
            cell_in_info = {'cell':'oConv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size']*cell_info['cardinality'],
                        'kernel_size':3,'stride':1,'padding':1,'dilation':1,'groups':1,'bias':cell_info['bias']}
        elif(cell_info['mode']=='fc'):
            cell_in_info = {'cell':'oConv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size']*cell_info['cardinality'],
                        'kernel_size':1,'stride':1,'padding':0,'dilation':1,'groups':1,'bias':cell_info['bias']}
        elif(cell_info['mode']=='fc_down'):
            cell_in_info = {'cell':'oConv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size']*cell_info['cardinality'],
                        'kernel_size':1,'stride':2,'padding':0,'dilation':1,'groups':1,'bias':cell_info['bias']}
        cell['in'] = Cell(cell_in_info)
        cell['activation'] = Cell({'cell':'Activation','mode':cell_info['activation']})
        cell['normalization'] = Cell({'cell':'Normalization','input_size':cell_info['input_size'],'mode':cell_info['normalization']}) if(cell_info['order']=='before') else \
        Cell({'cell':'Normalization','input_size':cell_info['output_size'],'mode':cell_info['normalization']})
        self.coordinates = self.make_coordinates(cell_info['input_size'],cell_info['output_size'],cell_info['cardinality'],cell_info['sharing_rate'])
        return cell
    
    def make_coordinates(self,input_size,output_size,cardinality,sharing_rate):
        coordinates = []
        output_coordinates = torch.arange(output_size*cardinality,device=device).view(-1,1).chunk(cardinality,dim=0)
        sharing_pivot = round(len(output_coordinates[0])*sharing_rate)
        for i in range(cardinality):
            output_coordinates[i][:sharing_pivot] = output_coordinates[0][:sharing_pivot]
        output_coordinates = torch.stack(output_coordinates,dim=0)
        input_coordinates = torch.arange(input_size,device=device).view(1,1,-1).expand(cardinality,1,-1)
        coordinates = [output_coordinates,input_coordinates]
        return coordinates
        
    def forward(self, input):
        x = input
        coordinates = [self.coordinates[0][config.PARAM['cardinality']],self.coordinates[1][config.PARAM['cardinality']]]
        if(self.cell_info['order']=='before'):
            x = self.cell['in'](self.cell['activation'](self.cell['normalization'](x)),coordinates)
        elif(self.cell_info['order']=='after'):
            x = self.cell['activation'](self.cell['normalization'](self.cell['in'](x,coordinates)))
        else:
            raise ValueError('wrong order')
        return x

class CartesianResBasicCell(nn.Module):
    def __init__(self, cell_info):
        super(CartesianResBasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(cell_info['mode'] == 'down'):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'CartesianBasicCell','mode':'fc_down',
                'normalization':cell_info['normalization'],'activation':'none','cardinality':cell_info['cardinality'],'sharing_rate':cell_info['sharing_rate']}
            elif(cell_info['input_size'] != cell_info['output_size']):
                cell_shortcut_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'CartesianBasicCell','mode':'fc',
                'normalization':cell_info['normalization'],'activation':'none','cardinality':cell_info['cardinality'],'sharing_rate':cell_info['sharing_rate']}
            else:
                cell_shortcut_info = {'cell':'none'}
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['output_size'],'cell':'CartesianBasicCell','mode':cell_info['mode'],
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'cardinality':cell_info['cardinality'],'sharing_rate':cell_info['sharing_rate']}
            cell_out_info = {'input_size':cell_info['output_size'],'output_size':cell_info['output_size'],'cell':'CartesianBasicCell','mode':'pass',
            'normalization':cell_info['normalization'],'activation':'none','cardinality':cell_info['cardinality'],'sharing_rate':cell_info['sharing_rate']}
            cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell[i]['activation'] = Cell({'cell':'Activation','mode':cell_info['activation']})
            cell_info['input_size'] = cell_info['output_size']
            cell_info['mode'] = 'pass'
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = self.cell[i]['shortcut'](x)
            x = self.cell[i]['in'](x)
            x = self.cell[i]['out'](x)
            x = self.cell[i]['activation'](x+shortcut)
        return x

class CartesianDenseCell(nn.Module):
    def __init__(self, cell_info):
        super(CartesianDenseCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['cardinality']*cell_info['bottleneck']*cell_info['growth_rate'],'cell':'CartesianBasicCell','mode':'fc',
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'cardinality':cell_info['cardinality'],'sharing_rate':cell_info['sharing_rate'],'order':'before'}
            cell_out_info = {'input_size':cell_info['cardinality']*cell_info['bottleneck']*cell_info['growth_rate'],'output_size':cell_info['cardinality']*cell_info['growth_rate'],'cell':'CartesianBasicCell','mode':'pass',
            'normalization':cell_info['normalization'],'activation':cell_info['activation'],'cardinality':cell_info['cardinality'],'sharing_rate':cell_info['sharing_rate'],'order':'before'}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell_info['input_size'] = cell_info['input_size'] + cell_info['cardinality']*cell_info['growth_rate']
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = list(x.chunk(self.cell_info['cardinality'],dim=1))
            x = self.cell[i]['in'](x)
            x = self.cell[i]['out'](x)
            x = list(x.chunk(self.cell_info['cardinality'],dim=1))
            for i in range(self.cell_info['cardinality']):
                x[i] = torch.cat([shortcut[i],x[i]],dim=1)
            x = torch.cat(x,dim=1)
        return x
        
class Cell(nn.Module):
    def __init__(self, cell_info):
        super(Cell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        if(self.cell_info['cell'] == 'none'):
            cell = nn.Sequential()
        elif(self.cell_info['cell'] == 'Normalization'):
            cell = Normalization(self.cell_info)
        elif(self.cell_info['cell'] == 'Activation'):
            cell = Activation(self.cell_info)
        elif(self.cell_info['cell'] == 'Conv2d'):
            default_cell_info = {'kernel_size':3,'stride':1,'padding':1,'dilation':1,'groups':1,'bias':False,'normalization':'none','activation':'relu'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = nn.Conv2d(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['kernel_size'],\
                self.cell_info['stride'],self.cell_info['padding'],self.cell_info['dilation'],self.cell_info['groups'],self.cell_info['bias'])
        elif(self.cell_info['cell'] == 'ConvTranspose2d'):
            default_cell_info = {'kernel_size':3,'stride':1,'padding':1,'output_padding':0,'dilation':1,'groups':1,'bias':False,'normalization':'none','activation':'relu'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = nn.ConvTranspose2d(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['kernel_size'],\
                self.cell_info['stride'],self.cell_info['padding'],self.cell_info['output_padding'],self.cell_info['groups'],self.cell_info['bias'],self.cell_info['dilation'])
        elif(self.cell_info['cell'] == 'oConv2d'):
            default_cell_info = {'kernel_size':3,'stride':1,'padding':1,'dilation':1,'groups':1,'bias':False}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = oConv2d(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['kernel_size'],\
                self.cell_info['stride'],self.cell_info['padding'],self.cell_info['dilation'],self.cell_info['groups'],self.cell_info['bias'])
        elif(self.cell_info['cell'] == 'BasicCell'):
            default_cell_info = {'mode':'pass','normalization':'none','activation':'relu','groups':1,'bias':False,'order':'after'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = BasicCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ResBasicCell'):
            cell = ResBasicCell(self.cell_info)
        elif(self.cell_info['cell'] == 'GroupResBasicCell'):
            cell = GroupResBasicCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ShuffleGroupResBasicCell'):
            cell = ShuffleGroupResBasicCell(self.cell_info)
        elif(self.cell_info['cell'] == 'BottleNeckCell'):
            cell = BottleNeckCell(self.cell_info)
        elif(self.cell_info['cell'] == 'GroupBottleNeckCell'):
            cell = GroupBottleNeckCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ShuffleGroupBottleNeckCell'):
            cell = ShuffleGroupBottleNeckCell(self.cell_info)
        elif(self.cell_info['cell'] == 'DenseCell'):
            cell = DenseCell(self.cell_info)
        elif(self.cell_info['cell'] == 'GroupDenseCell'):
            cell = GroupDenseCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ShuffleGroupDenseCell'):
            cell = ShuffleGroupDenseCell(self.cell_info)
        elif(self.cell_info['cell'] == 'LSTMCell'):
            default_cell_info = {'activation':'tanh'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = LSTMCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ResLSTMCell'):
            default_cell_info = {'activation':'tanh'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = ResLSTMCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ShuffleCell'):
            cell = ShuffleCell(self.cell_info)
        elif(self.cell_info['cell'] == 'PixelShuffleCell'):
            cell = PixelShuffleCell(self.cell_info)
        elif(self.cell_info['cell'] == 'PoolCell'):
            default_cell_info = {'kernel_size':2,'stride':None,'padding':0,'dilation':1,'return_indices':False,'ceil_mode':False,'count_include_pad':True}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = PoolCell(self.cell_info)
        elif(self.cell_info['cell'] == 'DownTransitionCell'):
            cell = DownTransitionCell(self.cell_info)
        elif(self.cell_info['cell'] == 'UpTransitionCell'):
            cell = UpTransitionCell(self.cell_info)
        elif(self.cell_info['cell'] == 'CartesianBasicCell'):
            default_cell_info = {'mode':'pass','normalization':'none','activation':'relu','cardinality':1,'sharing_rate':0,'bias':False,'order':'after'}
            self.cell_info = {**default_cell_info,**self.cell_info}            
            cell = CartesianBasicCell(self.cell_info)
        elif(self.cell_info['cell'] == 'CartesianResBasicCell'):
            cell = CartesianResBasicCell(self.cell_info)
        elif(self.cell_info['cell'] == 'CartesianDenseCell'):
            cell = CartesianDenseCell(self.cell_info)
        else:
            raise ValueError('model mode not supported')
        return cell
        
    def forward(self, *input):
        x = self.cell(*input)
        return x
