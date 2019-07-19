def generate_bash(namedict,batch_size,out_channels, kernel_size_cnn, stride_size_cnn, kernel_size_pool,stride_size_pool):
    commands = []
    for o in out_channels:
        out1=o[0]
        out2=o[1]
        for n in namedict.keys():
            for fft in namedict[n]:
                for b in batch_size:
                    for kc in kernel_size_cnn:
                        for sc in stride_size_cnn:
                            for kp in kernel_size_pool:
                                for sp in stride_size_pool:
                                    commands.append("python train_joint_spec_full_2d.py -n {} -fft {} -b {} -out1 {} -out2 {} -kc {} -sc {} -kp {} -sp {}".format(n,fft,b, out1,out2, kc,sc,kp,sp))
    with open('gpu_full_autogen_bash.sh','w+') as f:
        f.write('#!/bin/bash\n')
        for j in commands:
            f.write(j+"\n")

if __name__ == '__main__':
    namedict={"mel":[512,1024],"linear":[256,1024]}
    batch_size = [100]
    out_channels=[[64,16],[128,32],[256,16],[16,64],[32,128],[16,256]]
    kernel_size_cnn=[2,3,4]
    stride_size_cnn=[1]
    kernel_size_pool=[2,4]
    stride_size_pool=[2,4]
    generate_bash(namedict,batch_size,out_channels, kernel_size_cnn, stride_size_cnn, kernel_size_pool,stride_size_pool)
    print("Success")
