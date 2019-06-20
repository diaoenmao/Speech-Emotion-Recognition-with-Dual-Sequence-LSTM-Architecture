import argparse
import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader

from deep_model import GRUAudio, AttGRU, MeanPool, LSTM_Audio, ATT, Mean_Pool_2
from attention import AttGRU, MeanPool
from lstm_audio import LSTM_Audio
from process_audio_torch import IEMOCAP, my_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_parser():
    parser = argparse.ArgumentParser(description='Train and test your model as specified by the parameters you enter')
    parser.add_argument('--dataset', '-d', default='IEMOCAP', type=str,
                        help='IEMOCAP or Berlin (Berlin support still coming). IEMOCAP is default', dest='dataset')
    parser.add_argument('--model', '-m', default='gru', type=str, help='GRUAudio, AttGRU, MeanPool, Mean_Pool_2, LSTM_Audio, ATT', dest='model')
    parser.add_argument('--num_layers', '-nl', default=2, type=int, dest='num_layers')
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
    parser.add_argument('-no_train', default=True, action='store_false', help='Include to not train model',
                        dest='train')
    parser.add_argument('-no_test', default=True, action='store_false', help='Include to not test model', dest='test')
    parser.add_argument('-test_checkpoints', '-tc', default=False, action='store_true',
                        help='Include to test checkpoints', dest='test_checkpoints')
    parser.add_argument('-model_path', '-mp', default='', help='Include to specify path to model', dest='model_path')
    # parser.add_argument('--num_folds', default=10, type=int, dest='num_folds')
    return parser.parse_args()


def train_model(args, model_path, stats_path):
    model = get_model(args)
    model.cuda()
    model.train()

    # Use Adam as the optimizer with learning rate 0.01 to make it fast for testing purposes
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=0.3, patience=8, threshold=1e-3)

    # Load the training data
    training_data = IEMOCAP(train=True)
    train_loader = DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate,
                              num_workers=0)
    testing_data = IEMOCAP(train=False)
    test_loader = DataLoader(dataset=testing_data, batch_size=256, shuffle=True, collate_fn=my_collate, num_workers=0)

    best_test_acc=[]

    for epoch in range(args.num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        print("===================================" + str(epoch) + "==============================================")
        losses = 0
        correct=0
        losses_test = 0
        correct_test = 0
        model.train()
        for j, (input, target, seq_length) in enumerate(train_loader):
            
            input = pad_sequence(sequences=input, batch_first=True)

            input = pack_padded_sequence(input, lengths=seq_length, batch_first=True, enforce_sorted=False)        
            model.zero_grad()
            out, loss = model(input, target, seq_length=seq_length)
            losses += loss.item() * target.shape[0]
            loss.backward()
            optimizer.step()

            index = torch.argmax(out, dim=1)
            target_index = torch.argmax(target, dim=1).to(device)
            correct += sum(index == target_index).item()

        accuracy=correct*1.0/len(training_data)
        losses=losses / len(training_data)

        #after training
        model.eval()
        for test_case, target, seq_length in test_loader:
            test_case = pad_sequence(sequences=test_case, batch_first=True)
            test_case = pack_padded_sequence(test_case, lengths=seq_length, batch_first=True, enforce_sorted=False)
            out, loss = model(test_case, target, train=False, seq_length=seq_length)
            index = torch.argmax(out, dim=1)
            target_index = torch.argmax(target, dim=1).to(device)
            losses_test += loss.item() * index.shape[0]
            correct_test += sum(index == target_index).item()
        accuracy_test = correct_test * 1.0 / len(testing_data)
        best_test_acc.append(accuracy_test)
        losses_test = losses_test / len(testing_data)

        print("Training Loss: {} -------- Testing Loss: {} -------- Training Acc: {} -------- Testing Acc: {}".format(losses,losses_test, accuracy, accuracy_test))
        
        scheduler.step(losses)

        if (epoch + 1) % 5 == 0:
            checkpoint_path = build_model_path(args, True, epoch+1)
            torch.save(model.state_dict(), checkpoint_path)
        with open(stats_path, 'a+') as f:
            f.write("========================== Batch Normalization ===========================================")
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(args.dataset, args.hidden_dim, args.dr, args.num_epochs,
                                                              args.batch_size, args.bidirectional, args.lr,
                                                              args.num_layers,args.model, epoch, losses, losses_test, accuracy, accuracy_test))
            f.write("\n")
            f.write("================================="+"Best Test Accuracy"+str(max(best_test_acc))+"=====================================")

    torch.save(model.state_dict(), model_path)



def get_model(args):
    if args.model == 'GRUAudio':
        return GRUAudio(num_features=39, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout_rate=args.dr,
                     num_labels=4, batch_size=args.batch_size, bidirectional=args.bidirectional)
    elif args.model == 'AttGRU':
        return AttGRU(num_features=39, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout_rate=args.dr,
                        num_labels=4, batch_size=args.batch_size, bidirectional=args.bidirectional)
    elif args.model == 'MeanPool':
        return MeanPool(num_features=39, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout_rate=args.dr,
                        num_labels=4, batch_size=args.batch_size, bidirectional=args.bidirectional)
    elif args.model == 'LSTM_Audio':
        return LSTM_Audio(num_features=39, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout_rate=args.dr,
                        num_labels=4, batch_size=args.batch_size, bidirectional=args.bidirectional)
    elif args.model=="ATT":
        return ATT(num_features=39, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout_rate=args.dr,
                        num_labels=4, batch_size=args.batch_size, bidirectional=args.bidirectional)
    else:
        return Mean_Pool_2(num_features=39, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout_rate=args.dr,
                        num_labels=4, batch_size=args.batch_size, bidirectional=args.bidirectional)


def build_model_path(args, checkpoint=False, check_number=0):
    if checkpoint:
        return '/scratch/speech/models/classification/{}_hd_{}_dr_{}_e_{}_bs_{}_bi_{}_lr_{}_chkpt_{}_m_{}.pt'.format(
            args.dataset,
            args.hidden_dim,
            args.dr,
            args.num_epochs,
            args.batch_size,
            args.bidirectional,
            args.lr,
            check_number,
            args.model)
    return '/scratch/speech/models/classification/{}_hd_{}_dr_{}_e_{}_bs_{}_bi_{}_lr_{}_nl_{}_m_{}.pt'.format(
        args.dataset,
        args.hidden_dim,
        args.dr,
        args.num_epochs,
        args.batch_size,
        args.bidirectional,
        args.lr,
        args.num_layers,
        args.model)


if __name__ == '__main__':
    args = init_parser()
    if args.model_path == '':
        model_path = build_model_path(args)
    else:
        model_path = args.model_path
    if args.train:
        train_model(args, model_path)
   