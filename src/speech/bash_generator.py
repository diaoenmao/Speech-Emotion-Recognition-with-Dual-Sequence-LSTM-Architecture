def generate_bash(cuda, models, dropout_rates, batch_sizes, learning_rates, epochs, hidden_dims, bidirections):
    commands = []
    for model in models:
        for dr in dropout_rates:
            for b in batch_sizes:
                for lr in learning_rates:
                    for e in epochs:
                        for hd in hidden_dims:
                            for bi in bidirections:
                                commands.append(
                                    "python full_train_and_test.py -m {} -dr {} -b {} -lr {} -e {} -hd {} -tc {}".format(
                                        model, dr, b, lr, e, hd, bi))
    for i, device in enumerate(cuda):
        with open('gpu{}_autogen_bash.sh'.format(device),'w+') as f:
            f.write('#!/bin/bash\n')
            for j in range(int(len(commands) * 1.0 * i / len(cuda)),int(len(commands) * 1.0 * (i+1) / len(cuda))):
                f.write("CUDA_VISIBLE_DEVICES=\"{}\" {}\n".format(device, commands[j]))

if __name__ == '__main__':
    cuda = ["0","1","2","3"]
    models = ['Mean_Pool_2',"ATT"]
    dropout_rates = [0.0, 0.2, 0.4,0.8]
    batch_sizes = [128]
    learning_rates = [0.001]
    epochs = [150]
    hidden_dims = [200, 250, 300]
    bidirections = ['-bi', '']
    generate_bash(cuda, models, dropout_rates, batch_sizes, learning_rates, epochs, hidden_dims, bidirections)
    print("Success")
