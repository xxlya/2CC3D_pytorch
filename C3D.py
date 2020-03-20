import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self,window,nb_filter,batch_sz):
        super(MyNet,self).__init__()
        self.window = window
        self.nb_filter = nb_filter
        self.b_sz = batch_sz


        self.conv1 = nn.Conv3d(self.window, self.nb_filter[0], kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)
        self.bn1 = nn.BatchNorm3d(self.nb_filter[0])
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))


        self.conv2 = nn.Conv3d( self.nb_filter[0],  self.nb_filter[1], kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)
        self.bn2 = nn.BatchNorm3d(self.nb_filter[1])

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d( self.nb_filter[1], self.nb_filter[1], kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)
        self.bn3a = nn.BatchNorm3d(self.nb_filter[1])
        self.conv3b = nn.Conv3d(self.nb_filter[1],  self.nb_filter[2], kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)
        self.bn3b = nn.BatchNorm3d(self.nb_filter[2])
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d( self.nb_filter[2], self.nb_filter[2], kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)
        self.bn4a = nn.BatchNorm3d(self.nb_filter[2])
        self.conv4b = nn.Conv3d( self.nb_filter[2],  self.nb_filter[3], kernel_size=(3, 3, 3), padding=(1, 1, 1),bias=False)
        self.bn4b = nn.BatchNorm3d(self.nb_filter[3])
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc5 = nn.Linear(2*3*2*nb_filter[3], self.nb_filter[3])
        self.bn5 = nn.BatchNorm1d(self.nb_filter[3])
        self.fc6 = nn.Linear(self.nb_filter[3], self.nb_filter[4])
        self.bn6 = nn.BatchNorm1d( self.nb_filter[4])
        self.fc7 = nn.Linear(self.nb_filter[4], 2)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.65)

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax()



    def forward(self,x): # 45*54*45
        # x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.pool1(x) # 22*27*22
        x = self.dropout1(x)

        # x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.pool2(x) # 11*13*11
        x = self.dropout1(x)

        # x = self.relu(self.bn3a(self.conv3a(x)))
        # x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.bn3a(self.relu(self.conv3a(x)))
        x = self.bn3b(self.relu(self.conv3b(x)))
        x = self.pool3(x) # 5*6*5

        x = self.dropout2(x)

        # x = self.relu(self.bn4a(self.conv4a(x)))
        # x = self.relu(self.bn4b(self.conv4b(x)))
        x = self.bn4a(self.relu(self.conv4a(x)))
        x = self.bn4b(self.relu(self.conv4b(x)))
        x = self.pool4(x) # 2*3*2

        x = self.dropout2(x)

        x = x.view(-1, 2*3*2*self.nb_filter[3])
        # x = self.relu(self.bn5(self.fc5(x)))
        x = self.bn5(self.relu(self.fc5(x)))
        x = self.dropout1(x)
        # x = self.relu(self.bn6(self.fc6(x)))
        x = self.bn6(self.relu(self.fc6(x)))
        x = self.dropout1(x)
        res = self.softmax(self.fc7(x))

        return res

    '''
    #no  batchnormalization

        def forward(self,x): # 45*54*45 
            x = self.relu(self.conv1(x))
            x = self.pool1(x) # 22*27*22

            x = self.relu(self.conv2(x))
            x = self.pool2(x) # 11*13*11


            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x) # 5*6*5

            x = self.dropout(x)

            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x) # 2*3*2

            x = self.dropout(x)

            x = x.view(-1, 2*3*2*self.nb_filter[3])
            x = self.relu(self.fc5(x))
            x = self.dropout(x)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            res = self.softmax(self.fc7(x))

            return res
    '''

    def get_1x_lr_params(model):
        """
        This generator returns all the parameters for conv and two fc layers of the net.
        """
        b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
            model.fc5, model.fc6]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(model):
        """
        This generator returns all the parameters for the last fc layer of the net.
        """
        b = [model.fc7]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k
