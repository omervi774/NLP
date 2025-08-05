import argparse
import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np

batch_size = 32
output_size = 2
reload_model = False
learning_rate = 0.001
test_interval = 50

configs = {

        'MLP' : {
            'hidden_size' : 128,      
            'run_recurrent' : False,   
            'use_RNN' : False,  
        },

        'RNN' : {     
            'run_recurrent' : True,   
            'use_RNN' : True,         
        },

        'GRU' : {    
            'run_recurrent' : True,   
            'use_RNN' : False,         
        },

        'MLP_atten' : {      
            'run_recurrent' : False,   
            'use_RNN' : False,
            'atten_size' : 5         
        },
        
}

def parse_args():
    parser = argparse.ArgumentParser(description="Simple argument parser example")
    parser.add_argument("model_name", type=str, help="The name of the model to use")
    parser.add_argument("-hid", "--hidden_size", type=int, default=128, help="Size of the hidden layer (default: 128)")
    parser.add_argument("-a", "--atten_size", type=int, default=0, help="Size of the attention layer (default: 5)")
    parser.add_argument("-e", "--num_epochs", type=int, default=10, help="The numbers of epochs to run (default: 5)")
    args = parser.parse_args()
    return args

def get_run_parameters():
    args = parse_args()
    parameters = configs[args.model_name]
    parameters['hidden_size'] = args.hidden_size
    parameters['atten_size'] = args.atten_size
    parameters['num_epochs'] = args.num_epochs
    return parameters


# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)
class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias = True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels,out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,1,out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):        
        x = torch.matmul(x,self.matrix) 
        if self.use_bias:
            x = x+ self.bias 
        return x
        
# Implements RNN Unit
class ExRNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size) # fot the h_t
        self.hidden2out = nn.Linear(hidden_size, output_size) # fot the y_t

    def name(self):
        return f'RNN_{self.hidden_size}'

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), dim=1)  # [bs, input + hidden]
        hidden = self.sigmoid(self.in2hidden(combined))
        output = self.sigmoid(self.hidden2out(hidden))
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)

# Implements GRU Unit
class ExGRU(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.w_z = nn.Linear(hidden_size + input_size, hidden_size)
        self.w_r = nn.Linear(hidden_size + input_size, hidden_size)
        self.w = nn.Linear(hidden_size + input_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size) # fot the y_t

        
    def name(self):
        return f'GRU_{self.hidden_size}'

    def forward(self, x, hidden_state):
        # Implementation of GRU cell
        combined = torch.cat((x, hidden_state), dim=1)
        z_t = self.sigmoid(self.w_z(combined))
        r_t = self.sigmoid(self.w_r(combined))
        combined_tilde = torch.cat((r_t*hidden_state,x),dim=1)
        h_t_tilde = self.tanh(self.w(combined_tilde))
        hidden = (1 - z_t) * hidden_state + z_t * h_t_tilde
        output = self.sigmoid(self.hidden2out(hidden))
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class ExMLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()
        self.ReLU = torch.nn.ReLU()
        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.layer3 = MatMul(hidden_size, output_size)

    def name(self):
        return "MLP"

    def forward(self, x):
        # Token-wise MLP network implementation
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.layer3(x)
        return x


class ExRestSelfAtten(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, atten_size):
        super(ExRestSelfAtten, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)

        # Token-wise MLP + Restricted Attention network implementation
        self.layer1 = MatMul(input_size,hidden_size)
        self.atten_size = atten_size
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)

        # Final linear layer to get output logits (e.g., sentiment prediction)
        self.hidden_layer = MatMul(hidden_size, hidden_size)
        self.output_layer = MatMul(hidden_size, output_size)

        # Positional encodings (learned)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, hidden_size))


    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # Token-wise MLP + Restricted Attention network implementation
        x = self.layer1(x)
        x = self.ReLU(x)

        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends
        atten_size = self.atten_size
        padded = pad(x,(0,0,atten_size,atten_size,0,0))

        x_nei = []
        for k in range(-atten_size,atten_size+1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei,2)
        x_nei = x_nei[:,atten_size:-atten_size,:]
        
        Q = self.W_q(x).unsqueeze(2)  # [batch, seq_len, 1, hidden]
        K = self.W_k(x_nei)           # [batch, seq_len, 2w+1, hidden]
        V = self.W_v(x_nei)           # [batch, seq_len, 2w+1, hidden]

        # Step 4: Compute attention scores
        attn_logits = torch.sum(Q * K, dim=-1) / self.sqrt_hidden_size  # dot product, shape: [batch, seq_len, 2w+1]
        attn_weights = self.softmax(attn_logits)  # softmax over neighbors: [batch, seq_len, 2w+1]

        # Step 5: Compute weighted sum of values
        attended = torch.sum(V * attn_weights.unsqueeze(-1), dim=2)  # [batch, seq_len, hidden]
        
        # Step 6: Compute final token-wise scores
        x = self.hidden_layer(attended)
        x = self.ReLU(x)
        sub_score = self.output_layer(x)  # [batch, seq_len, output_size]

        return sub_score
