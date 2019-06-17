import argparse
import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader

from SE_audio_torch import GRUAudio
from process_audio_torch import IEMOCAP, my_collate
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_parser():
    parser = argparse.ArgumentParser(description='Train and test your model as specified by the parameters you enter')
    parser.add_argument('--dataset', '-d', default='IEMOCAP', type=str,
                        help='IEMOCAP or Berlin (Berlin support still coming). IEMOCAP is default', dest='dataset')
    parser.add_argument('--num_layers', '-nl',default=2, type=int, dest='num_layers')
    parser.add_argument('--hidden_dim', '-hd', default=200, type=int, dest='hidden_dim')
    parser.add_argument('-dropout_rate', '-dr', default=0.0, type=float,
                        help='Specify the dropout rate to use in range [0.0,1.0]. 0.0 is default', dest='dr')
    parser.add_argument('-batch_size', '-b', default=512, type=int,
                        help='Specifiy the batch size, should be power of 2. 512 is default', dest='batch_size')
    parser.add_argument('-learning_rate', '-lr', default=0.001, type=float,
                        help='Specify learning rate, 0.001 is default', dest='lr')
    parser.add_argument('-num_epochs', '-e', default=10, type=int, help='Specify the number of epochs. 10 is default',
                        dest='num_epochs')
    parser.add_argument('-bidirectional', '-bi', default=False, action='store_true',
                        help='Only include if you want to make it bidirectional', dest='bidirectional')
    # parser.add_argument('--num_folds', default=10, type=int, dest='num_folds')
    return parser.parse_args()


def train_model(args):
    model = GRUAudio(num_features=39, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout_rate=args.dr, num_labels=5,
                     batch_size=args.batch_size, bidirectional=args.bidirectional)
    model.cuda()

    # Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load the training data
    training_data = IEMOCAP(train=True)
    train_loader = DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate,
                              num_workers=0)

    for epoch in range(args.num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        print("===================================" + str(epoch) + "==============================================")
        losses = 0
        for j, (input, target, seq_length) in enumerate(train_loader):
            # print("==============================Batch " + str(j) + "=============================================")
            # pad input sequence to make all the same length
            # pdb.set_trace()

            input = pad_sequence(sequences=input, batch_first=True)

            # seq_length = seq_length.to(device)
            # make input a packed padded sequence
            #        pdb.set_trace()
            input = pack_padded_sequence(input, lengths=seq_length, batch_first=True, enforce_sorted=False)

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 3. Run our forward pass.
            out, loss = model(input, target)

            losses += loss.item()

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss.backward()
            optimizer.step()

        print("End of Epoch Loss: ", losses)
        # print(model.state_dict())
        if (epoch + 1) % 5 == 0:
            checkpoint_path = '/scratch/speech/models/classification/{}_hd_{}_dr_{}_e_{}_bs_{}_bi_{}_lr_{}_nl_{}_chkpt_{}.pt'.format(
                args.dataset, args.hidden_dim, args.dr, args.num_epochs, args.batch_size, args.bidirectional, args.lr, args.num_layers, str(epoch + 1))
            torch.save(model.state_dict(), checkpoint_path)

    model_path = '/scratch/speech/models/classification/{}_hd_{}_dr_{}_e_{}_bs_{}_bi_{}_lr_{}_nl_{}.pt'.format(args.dataset,
                                                                                                   args.hidden_dim,
                                                                                                   args.dr,
                                                                                                   args.num_epochs,
                                                                                                   args.batch_size,
                                                                                                   args.bidirectional,
                                                                                                   args.lr,
                                                                                                   args.num_layers)
    torch.save(model.state_dict(), model_path)


def test_model(args):
    model_path = '/scratch/speech/models/classification/{}_hd_{}_dr_{}_e_{}_bs_{}_bi_{}_lr_{}_nl_{}.pt'.format(args.dataset,
                                                                                                   args.hidden_dim,
                                                                                                   args.dr,
                                                                                                   args.num_epochs,
                                                                                                   args.batch_size,
                                                                                                   args.bidirectional,
                                                                                                   args.lr,
                                                                                                   args.num_layers)

    stats_path = '/scratch/speech/models/classification/{}_hd_{}_dr_{}_e_{}_bs_{}_bi_{}_lr_{}_nl_{}.txt'.format(args.dataset,
                                                                                                   args.hidden_dim,
                                                                                                   args.dr,
                                                                                                   args.num_epochs,
                                                                                                   args.batch_size,
                                                                                                   args.bidirectional,
                                                                                                   args.lr,
                                                                                                   args.num_layers)

    model = GRUAudio(num_features=39, hidden_dim=args.hidden_dim, num_layers=2, dropout_rate=args.dr, num_labels=5, batch_size=256, bidirectional=args.bidirectional)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    testing_data = IEMOCAP(train=False)
    test_loader = DataLoader(dataset=testing_data, batch_size=256, shuffle=True, collate_fn=my_collate, num_workers=0)
    print("Loading successful")

    correct = 0
    print(len(test_loader))
    print(len(testing_data))
    for test_case, target, seq_length in test_loader:
        test_case = pad_sequence(sequences=test_case, batch_first=True)
        test_case = pack_padded_sequence(test_case, lengths=seq_length, batch_first=True, enforce_sorted=False)
        out, loss = model(test_case, target, False)
        index = torch.argmax(out,dim=1)
        target_index=torch.argmax(target,dim=1)
        temp= target_index==index
        correct+=sum(temp).item()

    accuracy = correct * 1.0 / len(testing_data)
    print("accuracy:", accuracy)
    with open(stats_path, 'w+') as f:
        f.write("Accuracy of the model is: {}".format(accuracy))
        print(stats_path)


if __name__ == '__main__':
    cl_arguments = init_parser()
    train_model(cl_arguments)
    test_model(cl_arguments)
