def generate_bash(batch_size,out_channels, kernel_size_cnn, stride_size_cnn, kernel_size_pool,stride_size_pool):
    commands = []
    for o in out_channels:
        out1=o[0]
        out2=o[1]
        for b in batch_size:
            for kc in kernel_size_cnn:
                kc1=kc[0]
                kc2=kc[1]
                for sc in stride_size_cnn:
                    sc1=sc[0]
                    sc2=sc[1]
                    for kp in kernel_size_pool:
                        kp1=kp[0]
                        kp2=kp[1]
                        for sp in stride_size_pool:
                            commands.append("python train_joint_spec_multi.py -b {} -out1 {} -out2 {} -kc1 {} -kc2 {} -sc1 {} -sc2 {} -kp1 {} -kp2 {} -sp {}".format(b,out1,out2,kc1,kc2,sc1,sc1,kp1,kp2,sp))
    with open('gpu_full_autogen_bash.sh','w+') as f:
        f.write('#!/bin/bash\n')
        for j in commands:
            f.write(j+"\n")

if __name__ == '__main__':
    batch_size = [200]
    out_channels=[[16,64],[64,16]]
    kernel_size_cnn=[[4,2],[3,2],[5,3],[2,2],[3,3],[4,4],[2,1],[3,1],[4,1]]
    stride_size_cnn=[[1,1],[2,2],[2,1]]
    kernel_size_pool=[[2,2]]
    stride_size_pool=[2,2]
    generate_bash(batch_size,out_channels, kernel_size_cnn, stride_size_cnn, kernel_size_pool,stride_size_pool)
    print("Success")
