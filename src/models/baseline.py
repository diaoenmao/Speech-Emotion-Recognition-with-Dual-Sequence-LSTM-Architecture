import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules import Cell

device = config.PARAM['device']
   
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_info = config.PARAM['model']['encoder_info']
        return encoder_info

    def make_encoder(self):
        encoder = nn.ModuleList([])
        for i in range(len(self.encoder_info)):
            encoder.append(Cell(self.encoder_info[i]))
        return encoder
        
    def forward(self, input):
        x = input
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        return x
        
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_info = self.make_classifier_info()
        self.classifier = self.make_classifier()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.bias.data.zero_()
                
    def make_classifier_info(self):
        classifier_info = config.PARAM['model']['classifier_info']
        return classifier_info
        
    def make_classifier(self):
        classifier = nn.ModuleList([])
        for i in range(len(self.classifier_info)):
            classifier.append(Cell(self.classifier_info[i]))
        return classifier

    def classification_loss_fn(self, input, output):
        if(config.PARAM['loss_mode']['classification'] == 'ce'):
            loss_fn = F.cross_entropy
        else:
            raise ValueError('classification loss mode not supported')
        if(config.PARAM['tuning_param']['classification'] > 0):
            loss = loss_fn(output['classification'],input['label'],reduction='mean')
        else:
            loss = torch.tensor(0,device=device,dtype=torch.float32)
        return loss
        
    def forward(self, input):
        x = F.adaptive_avg_pool2d(input,1)
        x = self.classifier[0](x)
        x = x.view(x.size(0),-1)
        return x   
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, input):
        output = {'loss':torch.tensor(0,device=device,dtype=torch.float32),
                'classification':torch.tensor(0,device=device,dtype=torch.float32)}
                
        encoded = self.encoder(input['img'])
        output['classification'] = self.classifier(encoded)
        output['loss']  = self.classifier.classification_loss_fn(input,output)
        return output
    
def resnet14(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor]
    num_layers = [2,2,2]
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'num_layer':num_layers[0],'cell':'ResBasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'num_layer':num_layers[1],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'num_layer':num_layers[2],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[3],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def resnet14v2(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    base_size = int(model_TAG_list[3])
    encoder_size = [base_size,base_size,2*base_size,4*base_size]
    num_layers = [2,2,2]
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'num_layer':num_layers[0],'cell':'ResBasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'num_layer':num_layers[1],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'num_layer':num_layers[2],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[3],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def resnet18(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor,512*widen_factor]
    num_layers = [2,2,2,2]
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'num_layer':num_layers[0],'cell':'ResBasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'num_layer':num_layers[1],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'num_layer':num_layers[2],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[3],'output_size':encoder_size[4],'num_layer':num_layers[3],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}    
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[4],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def resnet29(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor]
    num_layers = [3,3,3]
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'num_layer':num_layers[0],'cell':'ResBasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'num_layer':num_layers[1],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'num_layer':num_layers[2],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[3],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def groupresnet29(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    groups = int(model_TAG_list[4])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor]
    num_layers = [3,3,3]
    transition_mode = 'avg'
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'num_layer':num_layers[0],'cell':'GroupResBasicCell','mode':'pass','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[1]},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'num_layer':num_layers[1],'cell':'GroupResBasicCell','mode':'pass','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[2]},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'num_layer':num_layers[2],'cell':'GroupResBasicCell','mode':'pass','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[3],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def shufflegroupresnet29(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    groups = int(model_TAG_list[4])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor]
    num_layers = [3,3,3]
    transition_mode = 'avg'
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'num_layer':num_layers[0],'cell':'ShuffleGroupResBasicCell','mode':'pass','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[1]},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'num_layer':num_layers[1],'cell':'ShuffleGroupResBasicCell','mode':'pass','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[2]},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'num_layer':num_layers[2],'cell':'ShuffleGroupResBasicCell','mode':'pass','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[3],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model
    
def resnet34(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor,512*widen_factor]
    num_layers = [3,4,6,3]
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'num_layer':num_layers[0],'cell':'ResBasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'num_layer':num_layers[1],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'num_layer':num_layers[2],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[3],'output_size':encoder_size[4],'num_layer':num_layers[3],'cell':'ResBasicCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}    
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[4],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def resnet50(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor,512*widen_factor]
    num_layers = [3,4,6,3]
    encoder_neck_size = [64,128,256,512]
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'neck_in_size':encoder_neck_size[0],'neck_out_size':encoder_neck_size[0],'num_layer':num_layers[0],'cell':'BottleNeckCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'neck_in_size':encoder_neck_size[1],'neck_out_size':encoder_neck_size[1],'num_layer':num_layers[1],'cell':'BottleNeckCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'neck_in_size':encoder_neck_size[2],'neck_out_size':encoder_neck_size[2],'num_layer':num_layers[2],'cell':'BottleNeckCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[3],'output_size':encoder_size[4],'neck_in_size':encoder_neck_size[3],'neck_out_size':encoder_neck_size[3],'num_layer':num_layers[3],'cell':'BottleNeckCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}    
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[4],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def resnet101(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor,512*widen_factor]
    num_layers = [3,4,23,3]
    encoder_neck_size = [64,128,256,512]
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'neck_in_size':encoder_neck_size[0],'neck_out_size':encoder_neck_size[0],'num_layer':num_layers[0],'cell':'BottleNeckCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'neck_in_size':encoder_neck_size[1],'neck_out_size':encoder_neck_size[1],'num_layer':num_layers[1],'cell':'BottleNeckCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'neck_in_size':encoder_neck_size[2],'neck_out_size':encoder_neck_size[2],'num_layer':num_layers[2],'cell':'BottleNeckCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[3],'output_size':encoder_size[4],'neck_in_size':encoder_neck_size[3],'neck_out_size':encoder_neck_size[3],'num_layer':num_layers[3],'cell':'BottleNeckCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}    
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[4],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def resnet152(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor,512*widen_factor]
    num_layers = [3,8,36,3]
    encoder_neck_size = [64,128,256,512]
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'neck_in_size':encoder_neck_size[0],'neck_out_size':encoder_neck_size[0],'num_layer':num_layers[0],'cell':'BottleNeckCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'neck_in_size':encoder_neck_size[1],'neck_out_size':encoder_neck_size[1],'num_layer':num_layers[1],'cell':'BottleNeckCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'neck_in_size':encoder_neck_size[2],'neck_out_size':encoder_neck_size[2],'num_layer':num_layers[2],'cell':'BottleNeckCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[3],'output_size':encoder_size[4],'neck_in_size':encoder_neck_size[3],'neck_out_size':encoder_neck_size[3],'num_layer':num_layers[3],'cell':'BottleNeckCell','mode':'down','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}    
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[4],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def resnext29(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    groups = int(model_TAG_list[4])
    groups_size = int(model_TAG_list[5])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor]
    num_layers = [3,3,3]
    encoder_neck_size = [groups*int(groups_size*(encoder_size[i+1]/(64.*widen_factor))) for i in range(3)]
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'neck_in_size':encoder_neck_size[0],'neck_out_size':encoder_neck_size[0],'num_layer':num_layers[0],'cell':'GroupBottleNeckCell','mode':'pass','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'neck_in_size':encoder_neck_size[1],'neck_out_size':encoder_neck_size[1],'num_layer':num_layers[1],'cell':'GroupBottleNeckCell','mode':'down','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'neck_in_size':encoder_neck_size[2],'neck_out_size':encoder_neck_size[2],'num_layer':num_layers[2],'cell':'GroupBottleNeckCell','mode':'down','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[3],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def shuffleresnext29(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    groups = int(model_TAG_list[4])
    groups_size = int(model_TAG_list[5])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor]
    num_layers = [3,3,3]
    encoder_neck_size = [groups*int(groups_size*(encoder_size[i+1]/(64.*widen_factor))) for i in range(3)]
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'neck_in_size':encoder_neck_size[0],'neck_out_size':encoder_neck_size[0],'num_layer':num_layers[0],'cell':'ShuffleGroupBottleNeckCell','mode':'pass','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'neck_in_size':encoder_neck_size[1],'neck_out_size':encoder_neck_size[1],'num_layer':num_layers[1],'cell':'ShuffleGroupBottleNeckCell','mode':'down','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'neck_in_size':encoder_neck_size[2],'neck_out_size':encoder_neck_size[2],'num_layer':num_layers[2],'cell':'ShuffleGroupBottleNeckCell','mode':'down','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[3],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model
    
def resnext50(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    groups = int(model_TAG_list[4])
    groups_size = int(model_TAG_list[5])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor,512*widen_factor]
    num_layers = [3,4,6,3]
    encoder_neck_size = [groups*int(groups_size*(encoder_size[i+1]/(64.*widen_factor))) for i in range(4)]    
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'neck_in_size':encoder_neck_size[0],'neck_out_size':encoder_neck_size[0],'num_layer':num_layers[0],'cell':'GroupBottleNeckCell','mode':'pass','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'neck_in_size':encoder_neck_size[1],'neck_out_size':encoder_neck_size[1],'num_layer':num_layers[1],'cell':'GroupBottleNeckCell','mode':'down','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'neck_in_size':encoder_neck_size[2],'neck_out_size':encoder_neck_size[2],'num_layer':num_layers[2],'cell':'GroupBottleNeckCell','mode':'down','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[3],'output_size':encoder_size[4],'neck_in_size':encoder_neck_size[3],'neck_out_size':encoder_neck_size[3],'num_layer':num_layers[3],'cell':'GroupBottleNeckCell','mode':'down','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[4],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def resnext101(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    widen_factor = int(model_TAG_list[3])
    groups = int(model_TAG_list[4])
    groups_size = int(model_TAG_list[5])
    encoder_size = [64,64*widen_factor,128*widen_factor,256*widen_factor,512*widen_factor]
    num_layers = [3,4,23,3]
    encoder_neck_size = [groups*int(groups_size*(encoder_size[i+1]/(64.*widen_factor))) for i in range(4)]    
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},   
        {'input_size':encoder_size[0],'output_size':encoder_size[1],'neck_in_size':encoder_neck_size[0],'neck_out_size':encoder_neck_size[0],'num_layer':num_layers[0],'cell':'GroupBottleNeckCell','mode':'pass','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[2],'neck_in_size':encoder_neck_size[1],'neck_out_size':encoder_neck_size[1],'num_layer':num_layers[1],'cell':'GroupBottleNeckCell','mode':'down','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[3],'neck_in_size':encoder_neck_size[2],'neck_out_size':encoder_neck_size[2],'num_layer':num_layers[2],'cell':'GroupBottleNeckCell','mode':'down','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[3],'output_size':encoder_size[4],'neck_in_size':encoder_neck_size[3],'neck_out_size':encoder_neck_size[3],'num_layer':num_layers[3],'cell':'GroupBottleNeckCell','mode':'down','groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[4],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','bias':True},
        ]
    model = Model()
    return model

def densenet86(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    bottleneck = int(model_TAG_list[3])
    growth_rate = int(model_TAG_list[4])
    growth_rates = [growth_rate,2*growth_rate,4*growth_rate]
    num_layers = [14,14,14]
    transition_mode = 'avg'   
    encoder_size = [2*growth_rate]
    for i in range(len(num_layers)):
        encoder_size.append(encoder_size[i]+growth_rates[i]*num_layers[i])
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none'},   
        {'input_size':encoder_size[0],'bottleneck':bottleneck,'growth_rate':growth_rates[0],'num_layer':num_layers[0],'cell':'DenseCell','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[1]},
        {'input_size':encoder_size[1],'bottleneck':bottleneck,'growth_rate':growth_rates[1],'num_layer':num_layers[1],'cell':'DenseCell','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[2]},
        {'input_size':encoder_size[2],'bottleneck':bottleneck,'growth_rate':growth_rates[2],'num_layer':num_layers[2],'cell':'DenseCell','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[3],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation'],'bias':True,'order':'before'},
        ]
    model = Model()
    return model

def groupdensenet86(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    bottleneck = int(model_TAG_list[3])
    groups = int(model_TAG_list[4])
    growth_rate = [8,16,32]
    num_layers = [14,14,14]
    transition_mode = 'avg'   
    encoder_size = [2*growth_rate[0]]
    for i in range(len(num_layers)):
        encoder_size.append(encoder_size[i]+growth_rate[i]*num_layers[i])
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none'},   
        {'input_size':encoder_size[0],'growth_rate':growth_rate[0],'num_layer':num_layers[0],'cell':'GroupDenseCell','bottleneck':bottleneck,'groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[1]},
        {'input_size':encoder_size[1],'growth_rate':growth_rate[1],'num_layer':num_layers[1],'cell':'GroupDenseCell','bottleneck':bottleneck,'groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[2]},
        {'input_size':encoder_size[2],'growth_rate':growth_rate[2],'num_layer':num_layers[2],'cell':'GroupDenseCell','bottleneck':bottleneck,'groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[3],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation'],'bias':True,'order':'before'},
        ]
    model = Model()
    return model

def shufflegroupdensenet86(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    bottleneck = int(model_TAG_list[3])
    groups = int(model_TAG_list[4])
    growth_rate = [8,16,32]
    num_layers = [14,14,14]
    transition_mode = 'avg'   
    encoder_size = [2*growth_rate[0]]
    for i in range(len(num_layers)):
        encoder_size.append(encoder_size[i]+growth_rate[i]*num_layers[i])
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none'},   
        {'input_size':encoder_size[0],'growth_rate':growth_rate[0],'num_layer':num_layers[0],'cell':'ShuffleGroupDenseCell','bottleneck':bottleneck,'groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[1]},
        {'input_size':encoder_size[1],'growth_rate':growth_rate[1],'num_layer':num_layers[1],'cell':'ShuffleGroupDenseCell','bottleneck':bottleneck,'groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[2]},
        {'input_size':encoder_size[2],'growth_rate':growth_rate[2],'num_layer':num_layers[2],'cell':'ShuffleGroupDenseCell','bottleneck':bottleneck,'groups':groups,'normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[3],'output_size':config.PARAM['classes_size'],'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation'],'bias':True,'order':'before'},
        ]
    model = Model()
    return model
    
def densenet121(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3
    bottleneck = int(model_TAG_list[3])
    growth_rate = [12,12,12,12]
    num_layers = [6,12,24,16]   
    transition_mode = 'avg'
    encoder_size = [2*growth_rate[0]]
    for i in range(len(num_layers)):
        if(i==0):
            encoder_size.append(encoder_size[i]+growth_rate[i]*num_layers[i])
        else:
            encoder_size.append(encoder_size[i]//2+growth_rate[i]*num_layers[i])
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':encoder_size[0],'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':'none','activation':'none'},   
        {'input_size':encoder_size[0],'bottleneck':bottleneck,'growth_rate':growth_rate[0],'num_layer':num_layers[0],'cell':'DenseCell','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[1],'output_size':encoder_size[1]//2,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation'],'order':'before'},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[1]//2},
        {'input_size':encoder_size[1]//2,'bottleneck':bottleneck,'growth_rate':growth_rate[1],'num_layer':num_layers[1],'cell':'DenseCell','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[2],'output_size':encoder_size[2]//2,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation'],'order':'before'},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[2]//2},
        {'input_size':encoder_size[2]//2,'bottleneck':bottleneck,'growth_rate':growth_rate[2],'num_layer':num_layers[2],'cell':'DenseCell','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':encoder_size[3],'output_size':encoder_size[3]//2,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation'],'order':'before'},
        {'cell':'DownTransitionCell','mode':transition_mode,'input_size':encoder_size[3]//2},
        {'input_size':encoder_size[4]//2,'bottleneck':bottleneck,'growth_rate':growth_rate[3],'num_layer':num_layers[3],'cell':'DenseCell','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        ]
    config.PARAM['model']['classifier_info'] = [
        {'input_size':encoder_size[4],'output_size':config.PARAM['classes_size'],'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation'],'bias':True,'order':'before'},
        ]
    model = Model()
    return model
   