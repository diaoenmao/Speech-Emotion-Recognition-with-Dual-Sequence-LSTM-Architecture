
def init():
    global PARAM
    PARAM = {
        'data_name': {'train':'CIFAR10','test':'CIFAR10'},
        'model_name': 'resnet18',
        'special_TAG': '1',
        'optimizer_name': 'SGD',
        'scheduler_name': 'CosineAnnealingLR',
        'lr': 1e-1,
        'weight_decay': 5e-4,
        'milestones': [150,225],
        'threshold': 5e-4,
        'threshold_mode': 'abs',
        'factor': 0.1,
        'normalize': True,
        'batch_size': {'train':128,'test':256},
        'num_workers': 0,
        'data_size': {'train':0,'test':0},
        'device': 'cuda',
        'num_epochs': 300,
        'save_mode': 0,
        'world_size': 1,
        'metric_names': {'train':['acc'],'test':['acc']},
        'topk': 1,
        'init_seed': 0,
        'num_Experiments': 1,
        'tuning_param': {'compression': 0, 'classification': 1},
        'loss_mode': {'compression':'mae','classification':'ce'},
        'normalization': 'bn',
        'activation': 'relu',
        'resume_mode': 0
    }