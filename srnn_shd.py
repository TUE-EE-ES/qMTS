from operator import is_
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import math
import torch.nn.functional as F
from torch.utils import data
import copy
import logging as _logging
import math
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

logger = _logging.getLogger(__name__)

torch.manual_seed(2)

train_X = np.load('data/trainX_4ms.npy')
train_y = np.load('data/trainY_4ms.npy').astype(np.float64)

test_X = np.load('data/testX_4ms.npy')
test_y = np.load('data/testY_4ms.npy').astype(np.float64)

logger.info(f"train dataset shape: {train_X.shape}")
logger.info(f"test dataset shape: {test_X.shape}")

batch_size = 64

tensor_trainX = torch.Tensor(train_X)  # transform to torch tensor
tensor_trainY = torch.Tensor(train_y)
train_dataset = data.TensorDataset(tensor_trainX, tensor_trainY)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
tensor_testX = torch.Tensor(test_X)  # transform to torch tensor
tensor_testY = torch.Tensor(test_y)
test_dataset = data.TensorDataset(tensor_testX, tensor_testY)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
STEP 2: MAKING DATASET ITERABLE
'''

decay = 0.1  # neuron decay rate
thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
num_epochs = 20  # 150  # n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

'''
STEP 3a: CREATE spike MODEL CLASS
'''

b_j0 = 0.01  # neural threshold baseline
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
alpha_bits = 8
rho_bits = 8
gamma_bits = 10

# define approximate firing function

gradient_type = 'MG'
logger.info(f"gradient_type: {gradient_type}")
scale = 6.
hight = 0.15
logger.info(f"height: {hight};scale: {scale}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger.info(f"device: {device}")

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


# define approximate firing function

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
  
        if gradient_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        elif gradient_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif gradient_type =='linear':
            temp = F.relu(1-input.abs())
        elif gradient_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * gamma

act_fun_adp = ActFun_adp.apply
# tau_m = torch.FloatTensor([tau_m])

def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dparams, params, dt=1, isAdapt=1):
    alpha = torch.exp(-1. * dt / tau_m).to(device)
    ro = torch.exp(-1. * dt / tau_adp).to(device)
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    if params is None:
        alpha_q = alpha
        ro_q = ro
        adp_const = beta * (1-ro)
        bj0 = b_j0
    else:
        alpha_q = torch.fake_quantize_per_tensor_affine(alpha, 2**(-alpha_bits), 0, 0, 2**(alpha_bits) - 1)
        ro_q = torch.fake_quantize_per_tensor_affine(ro, 2**(-rho_bits), 0, 0, 2**(rho_bits) - 1)
        roh = torch.mean(ro)
        adp_const = torch.round(beta*(1-roh)* 2**(gamma_bits))/2**(gamma_bits)
        bj0 = 0
    if dparams is None:
        b = ro_q * b + adp_const * spike
        mem = mem * alpha_q - b * spike * dt + (1 - alpha) * R_m * inputs 
    else:
        if dparams[3]:
            b = torch.fake_quantize_per_tensor_affine(b * ro_q + adp_const * spike, dparams[4], 0, 0, dparams[5] - 1) 
        else:
            b = ro_q * b + adp_const * spike
        if dparams[0]:
            # b = torch.fake_quantize_per_tensor_affine(b , dparams[2], 0, 0, dparams[3] - 1)
            mem = torch.fake_quantize_per_tensor_affine(mem * alpha_q, dparams[1], 0, -dparams[2], dparams[2] - 1)
            mem = torch.fake_quantize_per_tensor_affine(mem - b * spike * dt, dparams[1], 0, -dparams[2], dparams[2] - 1)
            mem = torch.fake_quantize_per_tensor_affine(mem + (1 - alpha) * R_m * inputs, dparams[1], 0, -dparams[2], dparams[2] - 1)         
        else:
            mem = mem * alpha_q - b * spike * dt + (1 - alpha) * R_m * inputs 

    
    inputs_ = mem - b - bj0

    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, b

def output_Neuron(inputs, mem, tau_m, dparams, params, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
    alpha = torch.exp(-1. * dt / tau_m).to(device)
    if params is None:
        alphaq = alpha
    else:
        alphaq = torch.fake_quantize_per_tensor_affine(alpha, 2**(-alpha_bits), 0, 0, 2**(alpha_bits) - 1)
    if dparams is None:
        mem = mem * alphaq + (1 - alpha) * R_m * inputs
    else:
        if dparams[0]:
            mem = torch.fake_quantize_per_tensor_affine(mem * alphaq, dparams[1], 0, -dparams[2], dparams[2] - 1)
            mem = torch.fake_quantize_per_tensor_affine(mem + (1 - alpha) * R_m * inputs, dparams[1], 0, -dparams[2], dparams[2] - 1)

        else:
            mem = mem * alphaq + (1 - alpha) * R_m * inputs
    return mem

class RNN_custom(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, norm_q = False, fixed_point = False, params = [None, None, None] , dparams = [None, None, None]):
        super(RNN_custom, self).__init__()

        self.hidden_size = hidden_size
        #self.output_size = output_size
        # self.hidden_size = input_size
        self.i_2_h1 = nn.Linear(input_size, hidden_size[0])
        self.h1_2_h1 = nn.Linear(hidden_size[0], hidden_size[0])
        self.h1_2_h2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.h2_2_h2 = nn.Linear(hidden_size[1], hidden_size[1])

        self.h2o = nn.Linear(hidden_size[1], output_size)

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(hidden_size[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(hidden_size[1]))
        self.tau_adp_o = nn.Parameter(torch.Tensor(output_size))
        self.tau_m_h1 = nn.Parameter(torch.Tensor(hidden_size[0]))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(hidden_size[1]))
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        # nn.init.orthogonal_(self.h1_2_h1.weight)
        # nn.init.orthogonal_(self.h2_2_h2.weight)
        nn.init.orthogonal_(self.h1_2_h1.weight)
        nn.init.orthogonal_(self.h2_2_h2.weight)
        nn.init.xavier_uniform_(self.i_2_h1.weight)
        nn.init.xavier_uniform_(self.h1_2_h2.weight)
        nn.init.xavier_uniform_(self.h2o.weight)

        nn.init.constant_(self.i_2_h1.bias, 0)
        nn.init.constant_(self.h1_2_h2.bias, 0)
        nn.init.constant_(self.h2_2_h2.bias, 0)
        nn.init.constant_(self.h1_2_h1.bias, 0)

        nn.init.normal_(self.tau_adp_h1,150,10)
        nn.init.normal_(self.tau_adp_h2, 150,10)
        nn.init.normal_(self.tau_adp_o, 150,10)
        nn.init.normal_(self.tau_m_h1, 20.,5)
        nn.init.normal_(self.tau_m_h2, 20.,5)
        nn.init.normal_(self.tau_m_o, 20.,5)

        self.dp = nn.Dropout(0.1)
        self.params = params
        self.dparams = dparams
        self.norm_q = norm_q
        self.fixed_point = fixed_point
    def weight_sparsity(self):
        t_w = torch.numel(self.i2h1_w) + torch.numel(self.h12h1_w) + torch.numel(self.h12h2_w) + torch.numel(self.h22h2_w) + torch.numel(self.h2o_w)
        tnz_w = torch.count_nonzero(self.i2h1_w) + torch.count_nonzero(self.h12h1_w) + torch.count_nonzero(self.h12h2_w) + torch.count_nonzero(self.h22h2_w) + torch.count_nonzero(self.h2o_w)
        logger.info(f"Desnsity input: {torch.count_nonzero(self.i2h1_w)/torch.numel(self.i2h1_w)}" )
        logger.info(f"Desnsity h12h1: {torch.count_nonzero(self.h12h1_w)/torch.numel(self.h12h1_w)}" )
        logger.info(f"Desnsity h12h2: {torch.count_nonzero(self.h12h2_w)/torch.numel(self.h12h2_w)}" )
        logger.info(f"Desnsity h22h2: {torch.count_nonzero(self.h22h2_w)/torch.numel(self.h22h2_w)}" )
        logger.info(f"Desnsity h2o: {torch.count_nonzero(self.h2o_w)/torch.numel(self.h2o_w)}" )
        logger.info(f"Density {tnz_w/t_w}")
    def quantize(self):
        if self.params[0] is None:
            self.i2h1_w = self.i_2_h1.weight.clone().detach()
            self.h12h1_w = self.h1_2_h1.weight.clone().detach()
            self.h12h2_w = self.h1_2_h2.weight.clone().detach()
            self.h22h2_w = self.h2_2_h2.weight.clone().detach()
            self.h2o_w = self.h2o.weight.clone().detach()
            self.h1_b = (self.i_2_h1.bias + self.h1_2_h1.bias).clone().detach()
            self.h2_b = (self.h1_2_h2.bias + self.h2_2_h2.bias).clone().detach()
            self.h2o_b = self.h2o.bias.clone().detach()
        else:
            h1_params, h2_params, o_params = self.params
            pos_w_max = []
            pos_b_max = []
            w_scale =           [h1_params[0], h2_params[0], o_params[0]]
            w_max_quant =       [h1_params[1], h2_params[1], o_params[1]]
            bias_scale =        [h1_params[2], h2_params[2], o_params[2]]
            bias_max_quant =    [h1_params[3], h2_params[3], o_params[3]]
            
            for w_max in w_max_quant:
                if w_max == 1:
                    pos_w_max.append(w_max)
                else:
                    pos_w_max.append(w_max-1)
            for b_max in bias_max_quant:
                if b_max == 1:
                    pos_b_max.append(b_max)
                else:
                    pos_b_max.append(b_max-1)                
            alpha1 = torch.exp(-1 * dt / self.tau_m_h1)
            alpha2 = torch.exp(-1 * dt / self.tau_m_h2)
            alpha3 = torch.exp(-1 * dt / self.tau_m_o)
            alpha1_s = alpha1.clone().detach()
            alpha2_s = alpha2.clone().detach()
            alpha3_s = alpha3.clone().detach()
            weight_scale_h1 = (w_scale[0] *(1./(1 - alpha1_s))).detach().requires_grad_(False)
            weight_scale_h2 = (w_scale[1] *(1./(1 - alpha2_s))).detach().requires_grad_(False)
            weight_scale_h3 = (w_scale[2] *(1./(1 - alpha3_s))).detach().requires_grad_(False)
            bias_scale_h1 = (bias_scale[0] *(1./(1 - alpha1_s))).detach().requires_grad_(False)
            bias_scale_h2 = (bias_scale[1] *(1./(1 - alpha2_s))).detach().requires_grad_(False)
            bias_scale_h3 = (bias_scale[2] *(1./(1 - alpha3_s))).detach().requires_grad_(False)
            #w_scale = [weight_scale_h1, weight_scale_h2, weight_scale_h3]
            self.i2h1_w  = torch.fake_quantize_per_channel_affine(self.i_2_h1.weight,  weight_scale_h1, torch.zeros(self.hidden_size[0], dtype=torch.float32).to(device), 0, -w_max_quant[0] , pos_w_max[0]).to(device)
            self.h12h1_w = torch.fake_quantize_per_channel_affine(self.h1_2_h1.weight, weight_scale_h1, torch.zeros(self.hidden_size[0], dtype=torch.float32).to(device), 0, -w_max_quant[0] , pos_w_max[0]).to(device)
            self.h12h2_w = torch.fake_quantize_per_channel_affine(self.h1_2_h2.weight, weight_scale_h2, torch.zeros(self.hidden_size[1], dtype=torch.float32).to(device), 0, -w_max_quant[1] , pos_w_max[1]).to(device)
            self.h22h2_w = torch.fake_quantize_per_channel_affine(self.h2_2_h2.weight, weight_scale_h2, torch.zeros(self.hidden_size[1], dtype=torch.float32).to(device), 0, -w_max_quant[1] , pos_w_max[1]).to(device)        
            self.h2o_w = torch.fake_quantize_per_channel_affine(self.h2o.weight, weight_scale_h3, torch.zeros(20, dtype=torch.float32).to(device), 0, -w_max_quant[2] , pos_w_max[2] - 1).to(device) 
            self.h1_b = torch.fake_quantize_per_channel_affine(self.i_2_h1.bias  + self.h1_2_h1.bias, bias_scale_h1, torch.zeros(self.hidden_size[0], dtype=torch.float32).to(device), 0, -bias_max_quant[0], pos_b_max[0]).to(device)
            self.h2_b = torch.fake_quantize_per_channel_affine(self.h1_2_h2.bias + self.h2_2_h2.bias, bias_scale_h2, torch.zeros(self.hidden_size[1], dtype=torch.float32).to(device), 0, -bias_max_quant[1], pos_b_max[1]).to(device)          
            self.h2o_b = torch.fake_quantize_per_channel_affine(self.h2o.bias, bias_scale_h3, torch.zeros(20, dtype=torch.float32).to(device), 0, -bias_max_quant[2], pos_b_max[2] - 1).to(device)        

            
    def norm_quantize(self):
        if self.params[0] is None:
            self.i2h1_w = self.i_2_h1.weight.clone().detach()
            self.h12h1_w = self.h1_2_h1.weight.clone().detach()
            self.h12h2_w = self.h1_2_h2.weight.clone().detach()
            self.h22h2_w = self.h2_2_h2.weight.clone().detach()
            self.h2o_w = self.h2o.weight.clone().detach()
            self.h1_b = (self.i_2_h1.bias + self.h1_2_h1.bias).clone().detach()
            self.h2_b = (self.h1_2_h2.bias + self.h2_2_h2.bias).clone().detach()
            self.h2o_b = self.h2o.bias.clone().detach()
        else:
            h1_params, h2_params, o_params = self.params
            pos_w_max = []
            pos_b_max = []
            w_max_quant = [h1_params[0], h2_params[0], o_params[0]]
            b_max_quant = [h1_params[1], h2_params[1], o_params[1]]
            w_range = [0, 0, 0]
            b_range = [0, 0, 0]
            i2h1_w = model.i_2_h1.weight.clone().detach()
            i2h1_w = torch.einsum('ji,j->ji', i2h1_w, 1. - torch.exp(-1 * dt / model.tau_m_h1))
            h12h1_w = model.h1_2_h1.weight.clone().detach()
            h12h1_w = torch.einsum('ji,j->ji', h12h1_w, (1 - torch.exp(-1 * dt / model.tau_m_h1)))
            i2h_range = torch.maximum(torch.max(i2h1_w), torch.abs(torch.min(i2h1_w)))
            h12h1_range = torch.maximum(torch.max(h12h1_w), torch.abs(torch.min(h12h1_w)))
            if(h12h1_range.item() > i2h_range.item()):
                w_range[0] = h12h1_range.item()
            else:
                w_range[0] = i2h_range.item()

            h12h2_w = model.h1_2_h2.weight.clone().detach()
            h12h2_w = torch.einsum('ji,j->ji', h12h2_w, 1. - torch.exp(-1 * dt / model.tau_m_h2))
            h22h2_w = model.h2_2_h2.weight.clone().detach()
            h22h2_w = torch.einsum('ji,j->ji', h22h2_w, (1 - torch.exp(-1 * dt / model.tau_m_h2)))
            h12h2_range = torch.maximum(torch.max(h12h2_w), torch.abs(torch.min(h12h2_w)))
            h22h2_range = torch.maximum(torch.max(h22h2_w), torch.abs(torch.min(h22h2_w)))
            if(h12h2_range.item() > h22h2_range.item()):
                w_range[1] = h12h2_range.item()
            else:
                w_range[1] = h22h2_range.item()

            i2h1_b = model.i_2_h1.bias.clone().detach()
            i2h1_b = torch.einsum('j,j->j', i2h1_b, 1. - torch.exp(-1 * dt / model.tau_m_h1))
            h12h1_b = model.h1_2_h1.bias.clone().detach()
            h12h1_b = torch.einsum('j,j->j', h12h1_b, (1 - torch.exp(-1 * dt / model.tau_m_h1)))
            h1_b = i2h1_b + h12h1_b
            h1b_range = torch.maximum(torch.max(h1_b), torch.abs(torch.min(h1_b)))
            b_range[0] = h1b_range.item()


            h12h2_b = model.h1_2_h2.bias.clone().detach()
            h12h2_b = torch.einsum('j,j->j', h12h2_b, 1. - torch.exp(-1 * dt / model.tau_m_h2))
            h22h2_b = model.h2_2_h2.bias.clone().detach()
            h22h2_b = torch.einsum('j,j->j', h22h2_b, (1 - torch.exp(-1 * dt / model.tau_m_h2)))
            h2_b = h12h2_b + h22h2_b
            h2b_range = torch.maximum(torch.max(h2_b), torch.abs(torch.min(h2_b)))
            b_range[1] = h2b_range.item()

            h2o_w = model.h2o.weight.clone().detach()
            h2o_w = torch.einsum('ji,j->ji', h2o_w, 1. - torch.exp(-1 * dt / model.tau_m_o))
            w_range[2] = torch.maximum(torch.max(h2o_w), torch.abs(torch.min(h2o_w))).item()
            h2o_b = model.h2o.bias.clone().detach()
            h2o_b = torch.einsum('j,j->j', h2o_b, 1. - torch.exp(-1 * dt / model.tau_m_o))
            b_range[2] = torch.maximum(torch.max(h2o_b), torch.abs(torch.min(h2o_b))).item()                
            for w_max in w_max_quant:
                if w_max == 1:
                    pos_w_max.append(w_max)
                else:
                    pos_w_max.append(w_max-1)
            for b_max in b_max_quant:
                if b_max == 1:
                    pos_b_max.append(b_max)
                else:
                    pos_b_max.append(b_max-1) 
            if self.fixed_point:
                for i, b_r in enumerate(b_range):
                    if b_r == 0:
                        b_range[i] = 2**(-5)
                for i, w_r in enumerate(w_range):
                    if w_r == 0:
                        w_range[i] = 2**(-5)
                w_range = [2**math.ceil(math.log2(w_r)) for w_r in w_range]
                b_range = [2**math.ceil(math.log2(w_r)) for w_r in b_range]

            w_scale = [w_r/w_m for w_r,w_m in zip(w_range, w_max_quant)]  
            b_scale = [w_r/w_m for w_r,w_m in zip(b_range, b_max_quant)]   
            alpha1 = torch.exp(-1 * dt / self.tau_m_h1)
            alpha2 = torch.exp(-1 * dt / self.tau_m_h2)
            alpha3 = torch.exp(-1 * dt / self.tau_m_o)
            alpha1_s = alpha1.clone().detach()
            alpha2_s = alpha2.clone().detach()
            alpha3_s = alpha3.clone().detach()
            weight_scale_h1 = (w_scale[0] *(1./(1 - alpha1_s))).detach().requires_grad_(False)
            weight_scale_h2 = (w_scale[1] *(1./(1 - alpha2_s))).detach().requires_grad_(False)
            weight_scale_h3 = (w_scale[2] *(1./(1 - alpha3_s))).detach().requires_grad_(False)
            bias_scale_h1 = (b_scale[0] *(1./(1 - alpha1_s))).detach().requires_grad_(False)
            bias_scale_h2 = (b_scale[1] *(1./(1 - alpha2_s))).detach().requires_grad_(False)
            bias_scale_h3 = (b_scale[2] *(1./(1 - alpha3_s))).detach().requires_grad_(False)
            #w_scale = [weight_scale_h1, weight_scale_h2, weight_scale_h3]
            self.i2h1_w  = torch.fake_quantize_per_channel_affine(self.i_2_h1.weight,  weight_scale_h1, torch.zeros(self.hidden_size[0], dtype=torch.float32).to(device), 0, -w_max_quant[0] , pos_w_max[0]).to(device)
            self.h12h1_w = torch.fake_quantize_per_channel_affine(self.h1_2_h1.weight, weight_scale_h1, torch.zeros(self.hidden_size[0], dtype=torch.float32).to(device), 0, -w_max_quant[0] , pos_w_max[0]).to(device)
            self.h12h2_w = torch.fake_quantize_per_channel_affine(self.h1_2_h2.weight, weight_scale_h2, torch.zeros(self.hidden_size[1], dtype=torch.float32).to(device), 0, -w_max_quant[1] , pos_w_max[1]).to(device)
            self.h22h2_w = torch.fake_quantize_per_channel_affine(self.h2_2_h2.weight, weight_scale_h2, torch.zeros(self.hidden_size[1], dtype=torch.float32).to(device), 0, -w_max_quant[1] , pos_w_max[1]).to(device)        
            self.h2o_w = torch.fake_quantize_per_channel_affine(self.h2o.weight, weight_scale_h3, torch.zeros(20, dtype=torch.float32).to(device), 0, -w_max_quant[2] , pos_w_max[2] - 1).to(device) 
            self.h1_b = torch.fake_quantize_per_channel_affine(self.i_2_h1.bias  + self.h1_2_h1.bias, bias_scale_h1, torch.zeros(self.hidden_size[0], dtype=torch.float32).to(device), 0, -b_max_quant[0], pos_b_max[0]).to(device)
            self.h2_b = torch.fake_quantize_per_channel_affine(self.h1_2_h2.bias + self.h2_2_h2.bias, bias_scale_h2, torch.zeros(self.hidden_size[1], dtype=torch.float32).to(device), 0, -b_max_quant[1], pos_b_max[1]).to(device)          
            self.h2o_b = torch.fake_quantize_per_channel_affine(self.h2o.bias, bias_scale_h3, torch.zeros(20, dtype=torch.float32).to(device), 0, -b_max_quant[2], pos_b_max[2] - 1).to(device)        


    def forward(self, input, verbose=False):
        batch_size, seq_num, input_dim = input.shape
        if self.params[0] is None:
            b_h1 = b_h2  = b_j0 * 1.8
        else:
            b_h1 = b_h2  = 0
        # mem_layer1 = torch.zeros(batch_size, self.hidden_size[0]).cuda()
        # mem_layer2 = torch.zeros(batch_size, self.hidden_size[1]).cuda()
        # mem_output = torch.zeros(batch_size, output_dim).cuda()
        mem_layer1 = torch.rand(batch_size, self.hidden_size[0]).cuda()
        mem_layer2 = torch.rand(batch_size, self.hidden_size[1]).cuda()
        mem_output = torch.rand(batch_size, output_dim).cuda()
        spike_layer1 = torch.zeros(batch_size, self.hidden_size[0]).cuda()
        spike_layer2 = torch.zeros(batch_size, self.hidden_size[1]).cuda()
        output = torch.zeros(batch_size, output_dim).cuda()


        hidden_spike_ = []
        hidden_spike2_ = []
        hidden_mem_ = [0, 0, 0]
        b_mem_ = [0, 0]

        if self.norm_q:
            self.norm_quantize()
        else:
            self.quantize()
        for i in range(seq_num):
            input_x = input[:, i, :]

            h_input = F.linear(input_x.float(), self.i2h1_w) + F.linear(spike_layer1, self.h12h1_w) + self.h1_b
            mem_layer1, spike_layer1, b_h1 = mem_update_adp(h_input, mem_layer1, spike_layer1, self.tau_adp_h1, b_h1,self.tau_m_h1, self.dparams[0], self.params[0])
            h2_input = F.linear(spike_layer1, self.h12h2_w) + F.linear(spike_layer2, self.h22h2_w) + self.h2_b
            mem_layer2, spike_layer2, b_h2 = mem_update_adp(h2_input, mem_layer2, spike_layer2, self.tau_adp_h2, b_h2, self.tau_m_h2, self.dparams[1], self.params[1])   
            o_input = F.linear(spike_layer2, self.h2o_w) + self.h2o_b
            mem_output = output_Neuron(o_input, mem_output, self.tau_m_o, self.dparams[2], self.params[2])

            if i > 10:
                output= output + F.softmax(mem_output, dim=1)#F.softmax(mem_output, dim=1)#
            if verbose:
                if(torch.max(torch.abs(mem_layer1)).item() > hidden_mem_[0]):
                    hidden_mem_[0] = torch.max(torch.abs(mem_layer1)).item()
                if(torch.max(torch.abs(mem_layer2)).item() > hidden_mem_[1]):
                    hidden_mem_[1] = torch.max(torch.abs(mem_layer2)).item()
                if(torch.max(torch.abs(mem_output)).item() > hidden_mem_[2]):
                    hidden_mem_[2] = torch.max(torch.abs(mem_output)).item()  
                if(torch.max(b_h1).item() > b_mem_[0]):
                    b_mem_[0] = torch.max(b_h1).item()
                if(torch.max(b_h2).item() > b_mem_[1]):
                    b_mem_[1] = torch.max(b_h2).item()
            hidden_spike_.append(spike_layer1.data.cpu().numpy().mean())            
            hidden_spike2_.append(spike_layer2.data.cpu().numpy().mean())            
        return output, hidden_spike_, hidden_spike2_, b_mem_
    def test(self, dataloader=test_loader,is_test=0):
        correct = 0
        total = 0
        mem_max = [0, 0, 0]
        max_B = [0, 0]
        fr0 = []
        fr1 = []
        frin = []
        # Iterate through test dataset
        for images, labels in dataloader:
            images = images.view(-1, seq_dim, input_dim).to(device)
            imgs = images.data.cpu().numpy().mean(axis=0)
            for img in imgs:
                frin.append(img)
            if is_test:
                outputs, fr0_,fr1_,b = self.forward(images, verbose=True)
                # mem_max = [max(m[i], mem_max[i]) for i in range(len(m))]
                # max_B = [max(b[i], max_B[i]) for i in range(len(b))]
            else:
                outputs, fr0_, fr1_, _ = self.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # total += labels.size(0)
            fr0x = fr0_
            for frx in fr0x:
                fr0.append(frx)
            fr1x = fr1_
            for frx in fr1x:
                fr1.append(frx)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.long().cpu()).sum()
            else:
                correct += (predicted == labels).sum()

        total = 8156
        accuracy = 100. * correct.numpy() / total
        
        logger.info(f"Mean input rate: {np.array(frin).mean()}")
        logger.info(f"Mean FR h1: {np.array(fr0).mean()}")
        logger.info(f"Mean FR h2: {np.array(fr1).mean()}")
        return accuracy, np.array(fr0).mean()
    def train(self, num_epochs=20, path='qspinn2', id='0'):
        acc = []
        best_accuracy = 80
        learning_rate =  1e-2
        criterion = nn.CrossEntropyLoss()
        if self.norm_quantize:
            if self.fixed_point:
                model_norm = "qspinn_fixedpt"
            else:
                model_norm = "qspinn_float"
        else:
            model_norm = "qmts"
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,eps=1e-5)
        scheduler = StepLR(optimizer, step_size=10, gamma=.5)  
        # for epoch in range(1,num_epochs):
        loss_sum = 0
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
            labels = labels.long().to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs, _,_,_ = self.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            # Calculate Loss: softmax --> cross entropy loss
            # loss = criterion(outputs, labels)
            # loss_sum+= loss
            # Getting gradients w.r.t. parameters
            # loss.backward()
            # Updating parameters
            # optimizer.step()

            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.long().cpu()).sum()
            else:
                correct += (predicted == labels).sum()

        scheduler.step()
        accuracy = 100. * correct.numpy() / total
        # accuracy,_ = test(model, train_loader)
        ts_acc,fr = self.test(is_test=0)
        # if ts_acc > best_accuracy and accuracy > 90:
        #     torch.save(self, './'+path+'/model_' + str(ts_acc) + '-readout-2layer-v2-4ms-'+id+'T.pth')
        #     best_accuracy = ts_acc

        # logger.info(f"epoch: {epoch}. Loss: {loss.item()}. Tr Accuracy: {accuracy}. Ts Accuracy:{ts_acc} . Fr: {fr}")
        logger.info(f"Tr Accuracy: {accuracy}. Ts Accuracy:{ts_acc} . Fr: {fr}")

        acc.append(accuracy)
            # if epoch %5==0:
            #     logger.info('epoch: ', epoch, '. Loss: ', loss_sum.item()/i, 
            #             '. Tr Accuracy: ', accuracy, '. Ts Accuracy: ', ts_acc,', Fr: ',fr)
        return acc

    def finetune(self, num_epochs=20, path='qspinn2', id = '0'):
        acc = []
        best_accuracy = 85

        learning_rate =  5e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,eps=1e-5)
        scheduler = StepLR(optimizer, step_size=10, gamma=.5)
        criterion = nn.CrossEntropyLoss()
        
        best_model = copy.deepcopy(self.state_dict())
        for epoch in range(1,num_epochs):
            loss_sum = 0
            total = 0
            correct = 0

            for i, (images, labels) in enumerate(train_loader):
                images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
                labels = labels.long().to(device)
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()
                # Forward pass to get output/logits
                outputs, _,_,_ = self.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)
                loss_sum+= loss
                # Getting gradients w.r.t. parameters
                loss.backward()
                # Updating parameters
                optimizer.step()

                total += labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.long().cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            scheduler.step()
            accuracy = 100. * correct.numpy() / total
            # accuracy,_ = test(model, train_loader)
            ts_acc,fr = self.test(is_test=0)
            if ts_acc > best_accuracy and accuracy > 0.90:
                torch.save(self, './' + path + '/model_' + str(ts_acc) + '-readout-2layer-v2-4m-'+id+'F.pth')
                best_accuracy = ts_acc
                best_model = copy.deepcopy(self.state_dict())#.state_dict()
            logger.info(f"epoch: {epoch}. Loss: {loss.item()}. Tr Accuracy: {accuracy}. Ts Accuracy:{ts_acc} . Fr: {fr}")


            acc.append(accuracy)
            # if epoch %5==0:
            #     logger.info('epoch: ', epoch, '. Loss: ', loss_sum.item()/i, 
            #             '. Tr Accuracy: ', accuracy, '. Ts Accuracy: ', ts_acc,', Fr: ',fr)
        #model.load_state_dict(model_dict)
        #model = copy.deepcopy(model_orig)
        # self.load_state_dict(best_model)
        return best_accuracy, best_model
        



def preprocess (model, fp32=False):
    i2h1_w = model.i_2_h1.weight
    i2h1_w = torch.einsum('ji,j->ji', i2h1_w, 1. - torch.exp(-1 * dt / model.tau_m_h1))
    h12h1_w = model.h1_2_h1.weight
    h12h1_w = torch.einsum('ji,j->ji', h12h1_w, (1 - torch.exp(-1 * dt / model.tau_m_h1)))
    if fp32:
        i2h_range = torch.maximum(torch.max(i2h1_w), torch.abs(torch.min(i2h1_w)))
        h12h1_range = torch.maximum(torch.max(h12h1_w), torch.abs(torch.min(h12h1_w)))
    else:
        i2h_range = torch.ceil(torch.log2(torch.maximum(torch.max(i2h1_w), torch.abs(torch.min(i2h1_w)))))
        h12h1_range = torch.ceil(torch.log2(torch.maximum(torch.max(h12h1_w), torch.abs(torch.min(h12h1_w)))))
    if(h12h1_range.item() > i2h_range.item()):
        l1_range = h12h1_range.item()
    else:
        l1_range = i2h_range.item()

    h12h2_w = model.h1_2_h2.weight
    h12h2_w = torch.einsum('ji,j->ji', h12h2_w, 1. - torch.exp(-1 * dt / model.tau_m_h2))
    h22h2_w = model.h2_2_h2.weight
    h22h2_w = torch.einsum('ji,j->ji', h22h2_w, (1 - torch.exp(-1 * dt / model.tau_m_h2)))
    if fp32:
        h12h2_range = torch.maximum(torch.max(h12h2_w), torch.abs(torch.min(h12h2_w)))
        h22h2_range = torch.maximum(torch.max(h22h2_w), torch.abs(torch.min(h22h2_w)))
    else:
        h12h2_range = torch.ceil(torch.log2(torch.maximum(torch.max(h12h2_w), torch.abs(torch.min(h12h2_w)))))
        h22h2_range = torch.ceil(torch.log2(torch.maximum(torch.max(h22h2_w), torch.abs(torch.min(h22h2_w)))))
    if(h12h2_range.item() > h22h2_range.item()):
        l2_range = h12h2_range.item()
    else:
        l2_range = h22h2_range.item()

    i2h1_b = model.i_2_h1.bias
    i2h1_b = torch.einsum('j,j->j', i2h1_b, 1. - torch.exp(-1 * dt / model.tau_m_h1))
    h12h1_b = model.h1_2_h1.bias
    h12h1_b = torch.einsum('j,j->j', h12h1_b, (1 - torch.exp(-1 * dt / model.tau_m_h1)))
    h1_b = i2h1_b + h12h1_b
    if fp32:
        h1b_range = torch.maximum(torch.max(h1_b), torch.abs(torch.min(h1_b)))
    else:
        h1b_range = torch.ceil(torch.log2(torch.maximum(torch.max(h1_b), torch.abs(torch.min(h1_b)))))
    b1_range = h1b_range.item()


    h12h2_b = model.h1_2_h2.bias
    h12h2_b = torch.einsum('j,j->j', h12h2_b, 1. - torch.exp(-1 * dt / model.tau_m_h2))
    h22h2_b = model.h2_2_h2.bias
    h22h2_b = torch.einsum('j,j->j', h22h2_b, (1 - torch.exp(-1 * dt / model.tau_m_h2)))
    h2_b = h12h2_b + h22h2_b
    if fp32:
        h2b_range = torch.maximum(torch.max(h2_b), torch.abs(torch.min(h2_b)))
    else:
        h2b_range = torch.ceil(torch.log2(torch.maximum(torch.max(h2_b), torch.abs(torch.min(h2_b)))))
    b2_range = h2b_range.item()

    h2o_w = model.h2o.weight
    h2o_w = torch.einsum('ji,j->ji', h2o_w, 1. - torch.exp(-1 * dt / model.tau_m_o))
    if fp32:
        h2o_range = torch.maximum(torch.max(h2o_w), torch.abs(torch.min(h2o_w))).item()
    else:
        h2o_range = torch.ceil(torch.log2(torch.maximum(torch.max(h2o_w), torch.abs(torch.min(h2o_w))))).item()
    h2o_b = model.h2o.bias
    h2o_b = torch.einsum('j,j->j', h2o_b, 1. - torch.exp(-1 * dt / model.tau_m_o))
    if fp32:
        h2ob_range = torch.maximum(torch.max(h2o_b), torch.abs(torch.min(h2o_b))).item()
    else:
        h2ob_range = torch.ceil(torch.log2(torch.maximum(torch.max(h2o_b), torch.abs(torch.min(h2o_b))))).item()    

    return [l1_range, l2_range, h2o_range], [b1_range, b2_range, h2ob_range]


'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 700
hidden_dim = [128,128]  # 128
output_dim = 20
seq_dim = 250  # Number of steps to unroll
num_encode = 700
total_steps = seq_dim





model2 = torch.load('./model_90.7-SHD-baseline.pth') #baseline model
model = RNN_custom(700, [128, 128], 20)
model.load_state_dict(model2.state_dict())
model.to(device)
print('Baseline accuracy fp32:')
print(model.test())
model.params = [0.0078125, 8, 0.0078125, 8], [0.015625, 8, 0.015625, 8], [0.001953125, 32, 0.0009765625, 8] #4-bit params
print('int4 params:', model.params)
print('Baseline accuracy int4:')
print(model.test())





model = torch.load('./model_92.2-SHD-Ternary') #ternary model
print('Ternary model params:', model.params)
model.to(device)
print('Ternary model accuracy:')
print(model.test())
