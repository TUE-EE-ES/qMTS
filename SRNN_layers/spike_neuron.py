import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

surrograte_type = 'MG'
print('gradient type: ', surrograte_type)

gamma = 0.5
lens = 0.5
R_m = 1


beta_value = 1.8
b_j0_value = 0.01
alpha_bits = 8
rho_bits = 8
gamma_bits = 10

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
        scale = 6.0
        hight = .15
        if surrograte_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        elif surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type =='linear':
            temp = F.relu(1-input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * gamma
    
    
act_fun_adp = ActFun_adp.apply    





def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dparams, params, dt=1, isAdapt=1,device=None):

    alpha = torch.exp(-1. * dt / tau_m).to(device)
    ro = torch.exp(-1. * dt / tau_adp).to(device)
    if isAdapt:
        beta = beta_value
    else:
        beta = 0.
    if params is None:
        alphaq = alpha
        roq = ro
        adaptation = beta * (1-ro)
        bj0 = b_j0_value
    else:       
        alphaq = torch.fake_quantize_per_tensor_affine(alpha, 2 ** (-alpha_bits), 0, 0, 2**(alpha_bits)-1).to(device)
        roq = torch.fake_quantize_per_tensor_affine(ro, 2 ** (-rho_bits), 0, 0, 2**(rho_bits)-1).to(device)
        ro_h = torch.mean(ro)#torch.nanmean(ro)
        adaptation = torch.round((1-ro_h)* beta* 2**(gamma_bits))/2**(gamma_bits)
        bj0 = 0

    if dparams[3]:
        b = torch.fake_quantize_per_tensor_affine( roq * b + adaptation * spike , dparams[4], 0, 0, dparams[5] - 1)
    else:
        b = roq * b + adaptation * spike
    if dparams[0]:
        mem = torch.fake_quantize_per_tensor_affine ( mem * alphaq, dparams[1], 0, -dparams[2], dparams[2] - 1)
        mem = torch.fake_quantize_per_tensor_affine ( mem - (b+bj0) * spike * dt, dparams[1], 0, -dparams[2], dparams[2] - 1)
        mem = torch.fake_quantize_per_tensor_affine (mem + (1 - alpha) * R_m * inputs, dparams[1], 0, -dparams[2], dparams[2] - 1)
    else:
        mem = mem * alphaq - (b+bj0) * spike * dt+ (1 - alpha) * R_m * inputs
 
    inputs_ = mem - (b+bj0) 
    spike = act_fun_adp(inputs_)  
    return mem, spike, b


def output_Neuron(inputs, mem, tau_m,dparams, params, dt=1,device=None):
    """
    The read out neuron is leaky integrator without spike
    """
    alpha = torch.exp(-1. * dt / tau_m).to(device)
    if params is None:
        alphaq = alpha
    else:
        alphaq = torch.fake_quantize_per_tensor_affine(alpha, 2**(-alpha_bits), 0, 0, 2**(alpha_bits) - 1)
    if dparams[0]:
        mem = torch.fake_quantize_per_tensor_affine(mem * alphaq, dparams[1], 0, -dparams[2], dparams[2] - 1)
    else:
        mem = mem * alphaq
    mem = mem + (1 - alpha) * R_m * inputs
    return mem
 