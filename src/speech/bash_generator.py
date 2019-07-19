def generate_bash(dataset,batch_size,out_channels, kernel_size_cnn, stride_size_cnn, kernel_size_pool,stride_size_pool):
    commands = []
    for d in dataset:
        for b in batch_size:
            for kc in kernel_size_cnn:
                for sc in stride_size_cnn:
                    for kp in kernel_size_pool:
                        for sp in stride_size_pool:
                            for out in out_channels:
                                commands.append("python train_joint_spec_full_2d.py -d {} -b {} -out {} -kc {} -sc {} -kp {} -sp {}".format(d, b, out, kc,sc,kp,sp))
    with open('gpu_full_autogen_bash.sh','w+') as f:
        f.write('#!/bin/bash\n')
        for j in commands:
            f.write(j+"\n")

if __name__ == '__main__':
    dataset= [['mel',512],['linear',256],['linear',512]]
    batch_size = [100]
    out_channels=[[64,16],[128,32],[256,16],[16,64],[32,128],[16,256]]
    kernel_size_cnn=[[3,3],[2,2],[4,4]]
    stride_size_cnn=[[1,1]]
    kernel_size_pool=[[2,2],[3,3],[4,4]]
    stride_size_pool=[[[2,2],[2,2]],[[4,4],[4,4]]]
    generate_bash(dataset,batch_size,out_channels, kernel_size_cnn, stride_size_cnn, kernel_size_pool,stride_size_pool)
    print("Success")
