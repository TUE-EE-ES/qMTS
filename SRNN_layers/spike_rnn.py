import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from SRNN_layers.spike_neuron import *#mem_update_adp
from SRNN_layers.spike_dense import *
     

class spike_rnn(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tauM = 20,tauAdp_inital =150, tau_initializer = 'normal',tauM_inital_std = 5,tauAdp_inital_std = 5,
                 is_adaptive=1,device='cpu',bias=True):
        super(spike_rnn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device

        self.dense = nn.Linear(input_dim,output_dim,bias=bias)
        self.recurrent = nn.Linear(output_dim,output_dim,bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_dim))
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
            nn.init.normal_(self.tau_adp,tauAdp_inital,tauAdp_inital_std)

        elif tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m,tauM,tauM_inital_std)
            self.tau_adp = multi_normal_initilization(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
    
    def parameters(self):
        return [self.dense.weight,self.recurrent.weight,self.dense.bias+self.recurrent.bias,self.tau_m,self.tau_adp]
    
    def quantize(self, params):
        
        self.params = params
        if params is None:
            self.dq = self.dense.weight.clone().detach()
            self.rq = self.recurrent.weight.clone().detach()
            self.bq = (self.dense.bias + self.recurrent.bias).clone().detach()
        else:
            w_scale, w_max, b_scale, b_max = params
            if(w_max == 1):
                w_pos = 1
            else:
                w_pos = w_max - 1
            if(b_max == 1):
                b_pos = 1
            else:
                b_pos = b_max - 1  
            alpha_s = torch.exp(-1 * 1 / self.tau_m).clone().detach()
            weight_scale_a = (w_scale *(1./(1 - alpha_s))).requires_grad_(False)     
            bias_scale_a = (b_scale *(1./(1 - alpha_s))).requires_grad_(False)     
            self.dq = torch.fake_quantize_per_channel_affine(self.dense.weight, weight_scale_a,  torch.zeros(self.output_dim).to(self.device), 0, -w_max, w_pos).to(self.device)
            self.rq = torch.fake_quantize_per_channel_affine(self.recurrent.weight, weight_scale_a,  torch.zeros(self.output_dim).to(self.device), 0, -w_max, w_pos).to(self.device)
            self.bq = torch.fake_quantize_per_channel_affine(self.dense.bias + self.recurrent.bias, bias_scale_a, torch.zeros(self.output_dim).to(self.device), 0, -b_max, b_pos).to(self.device)

    def dquantize(self, dparams):
        if dparams is None:
            mem_scale = None
            b_scale = None
            mem_max = None
            b_max = None
            flag_mem = False
            flag_b = False
        else:
            if dparams[0]:
                mem_scale = dparams[1]
                mem_max = dparams[2]
                flag_mem = True
            else:
                mem_scale = None
                mem_max = None
                flag_mem = False
            if dparams[3]:
                b_scale = dparams[4]
                b_max = dparams[5]
                flag_b = True
            else:
                b_scale = None
                b_max = None
                flag_b = False   
        self.dparams = [flag_mem, mem_scale, mem_max, flag_b, b_scale, b_max]
                    
    def set_neuron_state(self,batch_size):
        # self.mem = (torch.rand(batch_size,self.output_dim)*self.b_j0).to(self.device)
        #self.b_j0 = b_j0_value * beta_value
        self.mem = Variable(torch.zeros(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.zeros(batch_size,self.output_dim)).to(self.device)
        self.b = Variable(torch.zeros(batch_size,self.output_dim)).to(self.device)
    
    def forward(self,input_spike):
 
        d_input = F.linear(input_spike.float(), self.dq) + F.linear(self.spike, self.rq) + self.bq
        self.mem,self.spike,self.b = mem_update_adp(d_input, self.mem, self.spike, self.tau_adp, self.b,self.tau_m,  dparams = self.dparams, params = self.params, device=self.device, isAdapt=self.is_adaptive)             
        return  self.spike, self.mem, self.b

    def weight_sparsity(self):
        weight_sum = torch.count_nonzero(self.dq) + torch.count_nonzero(self.rq)
        #weight_sum = torch.count_nonzero(self.dense.weight) + torch.count_nonzero(self.recurrent.weight)
        #print(torch.count_nonzero(self.dense_q))
        #print(torch.count_nonzero(self.recurrent_q))
        return torch.count_nonzero(self.dq), torch.count_nonzero(self.rq)
