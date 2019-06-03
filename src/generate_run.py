import config
config.init()
import itertools

def main():
    gpu_ids = ['0','1','2','3']
    script_name = [['train_model.py']]
    model_names = [['resnet18']]
    init_seeds = [[0]]
    special_TAGs = [['1']]
    special_TAGs = list(itertools.product(*special_TAGs))
    special_TAGs = [['_'.join(special_TAGs[i]) for i in range(len(special_TAGs))]]
    controls = script_name + model_names + init_seeds + special_TAGs
    controls = list(itertools.product(*controls))
    s = '#!/bin/bash\n'
    for i in range(len(controls)):
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --model_name \\\'{}\\\' --init_seed {} --special_TAG \\\'{}\\\' &\n'.format(gpu_ids[i%len(gpu_ids)],*controls[i])        
    print(s)
    run_file = open("./run.sh", "w")
    run_file.write(s)
    run_file.close()
    exit()
    
if __name__ == '__main__':
    main()