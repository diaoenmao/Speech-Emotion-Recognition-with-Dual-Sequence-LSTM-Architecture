import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb



class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, kernel_size_pool=8, stride_pool=4):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride=1
        self.padding = int((kernel_size-1) / 2)
        self.kernel_size_pool=kernel_size_pool
        self.stride_pool=stride_pool

        self.Wxi = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whi = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxf = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whf = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding, bias=False)

        self.Wxc = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=True)
        self.Whc = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxo = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Who = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)
        
        self.max_pool = nn.MaxPool1d(self.kernel_size_pool, stride=self.stride_pool)
        self.batch = nn.BatchNorm1d(self.hidden_channels)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        ch_pool=self.batch(self.max_pool(ch))
        return ch_pool, ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape)).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape)).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape)).cuda()

        return (nn.Parameter(torch.zeros(batch_size, hidden, shape)).cuda(),
                nn.Parameter(torch.zeros(batch_size, hidden, shape)).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    # kernel size is also a list, same length as hidden_channels
    def __init__(self, input_channels, hidden_channels, kernel_size, step):
        super(ConvLSTM, self).__init__()
        assert len(hidden_channels)==len(kernel_size), "size mismatch"
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self._all_layers = []
        self.num_labels=4
        self.linear_dim=12
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classification = nn.Linear(self.linear_dim, self.num_labels)

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, target):
        # input should be a list of inputs, like a time stamp, maybe 1280 for 100 times. 
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[step]
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, shape = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=shape)
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_h, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (new_h, new_c)
            outputs.append(x)
        ## mean pooling and loss function
        out=[torch.unsqueeze(o, dim=3) for o in outputs]
        out=torch.flatten(torch.mean(torch.cat(out,dim=3),dim=3),start_dim=1)
        out = self.classification(out)

        loss = F.cross_entropy(out, torch.max(target, 1)[1].to(self.device))



        return torch.unsqueeze(out,dim=0), torch.unsqueeze(loss, dim=0)

