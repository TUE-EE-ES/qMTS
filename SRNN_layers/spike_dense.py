import numpy as np
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
from SRNN_layers.spike_neuron import *#mem_update_adp,output_Neuron
from torch.quantization import FakeQuantize, MovingAverageMinMaxObserver, MinMaxObserver


#w_scale = 0.25#-1 #0.25
# w = 1.3 -> clamp(round(1.3/0.25), -7, 7) -> 1.25 -> 5
# w = 2.7 -> 2.75 -> 11 (2.75/0.25) -> 7
            
    

class spike_dense(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tauM = 20,tauAdp_inital =150, tau_initializer = 'normal',tauM_inital_std = 5,tauAdp_inital_std = 5,
                 is_adaptive=1,device='cpu',bias=True):
        super(spike_dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device
        
        self.dense = nn.Linear(input_dim,output_dim,bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_dim))
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
            nn.init.normal_(self.tau_adp,tauAdp_inital,tauAdp_inital_std)

        elif tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m,tauM,tauM_inital_std)
            self.tau_adp = multi_normal_initilization(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
            #self.tau_adp_param = multi_normal_initilization(self.tau_adp_param,tauAdp_inital,1)
    
    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.tau_m,self.tau_adp]
    
    def set_neuron_state(self,batch_size):
        # self.mem = (torch.rand(batch_size,self.output_dim)*self.b_j0).to(self.device)
        #self.b_j0 = b_j0_value * beta_value
        self.mem = Variable(torch.zeros(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.zeros(batch_size,self.output_dim)).to(self.device)
        self.b = Variable(torch.zeros(batch_size,self.output_dim)).to(self.device)
        
    
    def quantize(self, params):
        self.params = params
        if self.params is None:
            self.wq = self.dense.weight.clone().detach()
            self.bq = self.dense.bias.clone().detach()
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
            self.wq = torch.fake_quantize_per_channel_affine(self.dense.weight,  weight_scale_a, torch.zeros(self.output_dim).to(self.device), 0, -w_max, w_pos).to(self.device)
            bias_scale_a = (b_scale *(1./(1 - alpha_s))).requires_grad_(False)    
            self.bq = torch.fake_quantize_per_channel_affine(self.dense.bias, bias_scale_a, torch.zeros(self.output_dim).to(self.device), 0, -b_max, b_pos).to(self.device)
   
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

    def forward(self,input_spike):
        if self.params is None:
            d_input = self.dense(input_spike.float()) 
        else:
            d_input = F.linear(input_spike.float(), self.wq) + self.bq
        self.mem,self.spike,self.b = mem_update_adp(d_input, self.mem, self.spike, self.tau_adp, self.b, self.tau_m, dparams = self.dparams, params = self.params, device=self.device, isAdapt=self.is_adaptive)#5.4  
        return self.spike, self.mem, self.b


    def weight_sparsity(self):
        weight_sum = torch.count_nonzero(self.wq)
        #weight_sum = torch.count_nonzero(self.dense.weight)
        return weight_sum

    
class readout_integrator(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tauM = 20,tau_initializer = 'normal',tauM_inital_std = 5,device='cpu',bias=True):
        super(readout_integrator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.dense = nn.Linear(input_dim,output_dim,bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
            
    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.tau_m]
    
    def set_neuron_state(self,batch_size):
        # self.mem = torch.rand(batch_size,self.output_dim).to(self.device)
        self.mem = (torch.zeros(batch_size,self.output_dim)).to(self.device)
    
    def quantize (self, params):

        self.params = params
        if params is None:
            self.wq = self.dense.weight.clone().detach()
            self.bq = self.dense.bias.clone().detach()
        else:
            w_scale, w_max, b_scale, b_max = params
            alpha_s = torch.exp(-1 * 1 / self.tau_m).clone().detach()
            weight_scale_a = (w_scale *(1./(1 - alpha_s))).requires_grad_(False)     
            bias_scale_a = (b_scale *(1./(1 - alpha_s))).requires_grad_(False)     
            self.wq = torch.fake_quantize_per_channel_affine(self.dense.weight, weight_scale_a,  torch.zeros(self.output_dim).to(self.device), 0, -w_max, w_max - 1).to(self.device)
            self.bq = torch.fake_quantize_per_channel_affine(self.dense.bias, bias_scale_a, torch.zeros(self.output_dim).to(self.device), 0, -b_max, b_max - 1).to(self.device)

    def dquantize(self, dparams):
        if dparams is None:
            mem_scale = None
            mem_max = None
            flag_mem = False
        else:
            if dparams[0]:
                mem_scale = dparams[1]
                mem_max = dparams[2]
                flag_mem = True
            else:
                mem_scale = None
                mem_max = None
                flag_mem = False
        self.dparams = [flag_mem, mem_scale, mem_max]
                          
    def forward(self,input_spike):
        if self.params is None:
            d_input = self.dense(input_spike.float())
        else:
            d_input = F.linear(input_spike.float(), self.weight_q, bias = self.bq)
        self.mem = output_Neuron(d_input,self.mem,self.tau_m, dparams = self.dparams, params = self.params, device=self.device)
        return self.mem
    def weight_sparsity(self):
        weight_sum = torch.count_nonzero(self.wq)
        #weight_sum = torch.count_nonzero(self.dense.weight)
        return weight_sum
def multi_normal_initilization(param, means=[10,200],stds = [5,20]):
    shape_list = param.shape
    if len(shape_list) == 1:
        num_total = shape_list[0]
    elif len(shape_list) == 2:
        num_total = shape_list[0]*shape_list[1]

    num_per_group = int(num_total/len(means))
    # if num_total%len(means) != 0: 
    num_last_group = num_total%len(means)
    a = []
    for i in range(len(means)):
        a = a+ np.random.normal(means[i],stds[i],size=num_per_group).tolist()
        if i == len(means):
            a = a+ np.random.normal(means[i],stds[i],size=num_per_group+num_last_group).tolist()
    p = np.array(a).reshape(shape_list)
    with torch.no_grad():
        param.copy_(torch.from_numpy(p).float())
    return param