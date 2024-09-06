import os
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ReLU

PATH_TO_TRAIN_IMGDIR = os.environ['PATH_TO_TRAIN_IMGDIR']
PATH_TO_TRAIN_LABELS = os.environ['PATH_TO_TRAIN_LABELS']

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):

        """
        Args:
        - nIn: input size
        - nHidden: hidden layer size 
        - nOut: output size 
        """

        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut) # nHidden * 2 because LSTM is bidirectional 

    def forward(self, input):

        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size() # time steps, batch size, hidden units
        t_rec = recurrent.view(T * b, h) # reshaping for linear input 
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1) # reshape back

        return output

class Model(nn.Module):

    def __init__(self, nHidden, num_classes):

        super(Model, self).__init__()
        
        # CNN for features extraction before passing to RNN
        self.conv0 = Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1 = Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = Conv2d(512, 512, kernel_size=(4, 4), stride=(1, 1))

        self.pool0 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
        self.pool5 = MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)

        self.bn2 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn4 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn6 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu = ReLU()

        # RNN 
        self.rnn = nn.Sequential(
            BidirectionalLSTM(nHidden*2, nHidden, nHidden),
            BidirectionalLSTM(nHidden, nHidden, num_classes))


    def forward(self, src):
        
        #CNN layers 
        x = self.pool0(self.relu(self.conv0(src)))
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool5(self.relu(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))

        b, c, h, w = x.size() # b-batch size, c-channels (512 from conv6), h-height, w-width (t-time steps)
        assert h == 1, "the height of conv must be 1 to pass to rnn" 
        x = x.squeeze(2) # [b, c, w] # remove the dimension (h, which is 1)
        x = x.permute(2, 0, 1)  # permute it to make suitable for the rnn

        logits = self.rnn(x) # [w, b, num_classes]
        output = torch.nn.functional.log_softmax(logits, 2) # map the outputs to probabilities 

        return output
    
