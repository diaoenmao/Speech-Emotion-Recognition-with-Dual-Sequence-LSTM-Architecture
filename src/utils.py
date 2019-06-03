import config
import collections.abc as container_abcs
import errno
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torchvision
from itertools import repeat
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.utils import make_grid
from torchvision.utils import save_image

def makedir_exist_ok(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return

def save(input,dir,protocol = 2,mode='torch'):
    dirname = os.path.dirname(dir)
    makedir_exist_ok(dirname)
    if(mode=='torch'):
        torch.save(input,dir,pickle_protocol=protocol)
    elif(mode=='numpy'):
        np.save(dir,input)
    else:
        raise ValueError('Not supported save mode')
    return

def load(dir,mode='torch'):
    if(mode=='torch'):
        return torch.load(dir, map_location=lambda storage, loc: storage)
    elif(mode=='numpy'):
        return np.load(dir)  
    else:
        raise ValueError('Not supported save mode')
    return                

def save_model(model, dir):
    dirname = os.path.dirname(dir)
    makedir_exist_ok(dirname)
    torch.save(model.state_dict(), dir)
    return
    
def load_model(model, dir):
    checkpoint = torch.load(dir)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:       
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model
    
def print_model(model):
    for p in model.parameters():
        print(p)
    return
        
def list_dir(root, prefix=False):
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)))
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories

def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)))
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files
    
def save_img(img,path,nrow=0,batch_first=False):
    if(img.dim()==4):
        dirname = os.path.dirname(path)
        makedir_exist_ok(dirname)
        if(nrow!=0):
            save_image(img,path,padding=0,nrow=nrow)
        else:
            save_image(img,path,padding=0)
    elif(img.dim()==5 and nrow!=0):
        dirname = os.path.dirname(path)
        makedir_exist_ok(dirname)
        seq_img = []
        if(batch_first==False):
            for i in range(img.size(1)):          
                seq_img.append(make_grid(img[:,i,],nrow=nrow))
        else:
            for i in range(img.size(0)):          
                seq_img.append(make_grid(img[i],nrow=nrow))
        img = torch.stack(seq_img,0)
        save_image(img,path,padding=0)
    else:
        raise ValueError('Not valid image to save')
    return

def dict_to_device(input,device):
    if(isinstance(input,dict)):
        for key in input:
            if(isinstance(input[key], list)):
                for i in range(len(input[key])):
                    input[key][i] = input[key][i].to(device)
            elif(isinstance(input[key], torch.Tensor)):
                input[key] = input[key].to(device)
            elif(isinstance(input[key], dict)):
                input[key] = dict_to_device(input[key], device)
            else:
                raise ValueError('input type not supported')
    else:
        input = input.to(device)
    return input
    
def pad_sequence(sequences, batch_first=False, padding_value=0):
    trailing_dims = sequences[0].size()[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    out_tensor = sequences[0].new(*out_dims).fill_(padding_value)
    lengths = sequences[0].new_zeros(len(sequences),dtype=torch.long)
    for i, tensor in enumerate(sequences):
        lengths[i] = tensor.size(0)
        if batch_first:
            out_tensor[i, :lengths[i], ...] = tensor
        else:
            out_tensor[:lengths[i], i, ...] = tensor

    return out_tensor, lengths
    
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse  
    
def apply_along_dim(input, *other_input, fn, dim, m='flat', **other_kinput):
    _tuple = _ntuple(2)
    dim = _tuple(dim)
    output = []
    if(m=='list'):
        for i, input_i in enumerate(torch.unbind(input, dim=dim[0])):
            cur_other_input = [x[i] for x in other_input]
            cur_other_kinput = {k:other_kinput[k][i] for k in other_kinput}
            output.append(fn(input_i,*cur_other_input,**cur_other_kinput))
    elif(m=='flat'):
        for i, input_i in enumerate(torch.unbind(input, dim=dim[0])):
            cur_other_input = other_input
            cur_other_kinput = other_kinput
            output.append(fn(input_i,*cur_other_input,**cur_other_kinput))  
    else:
        raise ValueError('Apply mode not supported')
    output = torch.stack(output, dim=dim[1])
    return output

def apply_fn(module,fn):
    for n, m in module.named_children():
        if(hasattr(m,fn)):
            exec('m.{0}()'.format(fn))
        if(sum(1 for _ in m.named_children())!=0):
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return

# ===================Function===================== 
def p_inverse(A):
    pinv = (A.t().matmul(A)).inverse().matmul(A.t())
    return pinv

def RGB_to_L(input):
    output = 0.2989*input[:,[0],]+0.5870*input[:,[1],]+0.1140*input[:,[2],]
    return output
    
def L_to_RGB(input):
    output = input.expand(input.size(0),3,input.size(2),input.size(3))
    return output

def gumbel_softmax(logits, tau=1, hard=False, sample=True, dim=-1):
    if(sample):
        eps = 1e-20
        U = torch.rand(logits.size(),device=logits.device)
        noise = -(torch.log(-torch.log(U + eps) + eps))
        gumbels = (logits + noise) / tau
    else:
        gumbels = logits / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.max(dim,keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim,index,1.0)
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        ret = y_soft
    return ret

def gumbel_softrank(logits, tau=1, hard=False, sample=True, dim=-1):
    if(sample):
        eps = 1e-20
        U = torch.rand(logits.size(),device=logits.device)
        noise = -(torch.log(-torch.log(U + eps) + eps))
        gumbels = (logits + noise) / tau
    else:
        gumbels = logits / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.topk(y_soft.size(dim),dim)[1].view(-1,1)
        y_hard = logits.new_zeros(logits.size(dim),logits.size(dim)).scatter_(dim, index, 1.0)
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        ret = y_soft
    return ret
    