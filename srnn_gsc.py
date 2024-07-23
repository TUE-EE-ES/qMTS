import os
import sys
sys.path.append("..")
import time
import numpy as np
import librosa
import scipy.io.wavfile as wav
# from tqdm import tqdm_notebook
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
import torchvision
from torch.optim.lr_scheduler import StepLR,MultiStepLR,LambdaLR,ExponentialLR
from data import SpeechCommandsDataset,Pad, MelSpectrogram, Rescale#,Normalize
from optim import RAdam
from utils import *
import pandas as pd
import torch.nn as nn
import math
import copy



dtype = torch.float
torch.manual_seed(0) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data_root = "gsc_v1_data/train"
test_data_root = "gsc_v1_data/test"
training_words = os.listdir(train_data_root)
training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root,x))]
training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root,x)) if x[0] != "_" ]
print("{} training words:".format(len(training_words)))
print(training_words)


testing_words = os.listdir(test_data_root)
testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x))]
testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x)) 
                 if x[0] != "_"]

print("{} testing words:".format(len(testing_words)))
print(testing_words)

label_dct = {k:i for i,k in enumerate(testing_words + ["_silence_", "_unknown_"])}
for w in training_words:
    label = label_dct.get(w)
    if label is None:
        label_dct[w] = label_dct["_unknown_"]

print("label_dct:")
print(label_dct)

sr = 16000
size = 16000

noise_path = os.path.join(train_data_root, "_background_noise_")
noise_files = []
for f in os.listdir(noise_path):
    if f.endswith(".wav"):
        full_name = os.path.join(noise_path, f)
        noise_files.append(full_name)
print("noise files:")
print(noise_files)

# generate silence training and validation data

silence_folder = os.path.join(train_data_root, "_silence_")
if not os.path.exists(silence_folder):
    os.makedirs(silence_folder)
    # 260 validation / 2300 training / 260 test
    generate_random_silence_files(2560, noise_files, size, os.path.join(silence_folder, "rd_silence"))    
    #generate_random_silence_files(2820, noise_files, size, os.path.join(silence_folder, "rd_silence"))    

    # save 260 files for validation / 260 test
    silence_files = [fname for fname in os.listdir(silence_folder)]
    with open(os.path.join(train_data_root, "silence_validation_list.txt"),"w") as f:
        f.writelines("_silence_/"+ fname + "\n" for fname in silence_files[:260])
    #with open(os.path.join(train_data_root, "silence_testing_list.txt"),"w") as f:
    #    f.writelines("_silence_/"+ fname + "\n" for fname in silence_files[260:520])

n_fft = int(30e-3*sr)
hop_length = int(10e-3*sr)
n_mels = 40
fmax = 4000
fmin = 20
delta_order = 2
stack = True

melspec = MelSpectrogram(sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order, stack=stack)
pad = Pad(size)
rescale = Rescale()
#normalize = Normalize()

transform = torchvision.transforms.Compose([pad,melspec,rescale])


def collate_fn(data):
    
    X_batch = np.array([d[0] for d in data])
    std = X_batch.std(axis=(0,2), keepdims=True)
    X_batch = torch.tensor(X_batch/std)
    y_batch = torch.tensor([d[1] for d in data])
    
    return X_batch, y_batch 

#batch_size = 32
batch_size = 64

train_dataset = SpeechCommandsDataset(train_data_root, label_dct, transform = transform, mode="train", max_nb_per_class=None)
train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights,len(train_dataset.weights))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=train_sampler, collate_fn=collate_fn)

valid_dataset = SpeechCommandsDataset(train_data_root, label_dct, transform = transform, mode="valid", max_nb_per_class=260)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

test_dataset = SpeechCommandsDataset(test_data_root, label_dct, transform = transform, mode="test")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

counts = np.zeros(12)
total = 0
for i, (images, labels) in enumerate(valid_dataloader):

    labels = labels.view((-1)).long().to("cpu")
    values, batch_counts = np.unique(labels, return_counts=True)
    this_counts = np.zeros(12)
    for i in range(len(values)):
        this_counts[values[i]] += batch_counts[i]
    total += np.sum(this_counts)
    counts += this_counts
weights = np.ones(12) - (counts/total)

counts = np.zeros(12)
total = 0
for i, (images, labels) in enumerate(test_dataloader):
    # if i ==0:

    labels = labels.view((-1)).long().to("cpu")
    values, batch_counts = np.unique(labels, return_counts=True)
    this_counts = np.zeros(12)
    for i in range(len(values)):
        this_counts[values[i]] += batch_counts[i]
    total += np.sum(this_counts)
    counts += this_counts
weights = np.ones(12) - (counts/total)


#####################################################################################################################3
# create network

from SRNN_layers.spike_dense import *
from SRNN_layers.spike_neuron import *
from SRNN_layers.spike_rnn import *
thr_func = ActFun_adp.apply  
is_bias=True
n = 300


class RNN_spike(nn.Module):
    def __init__(self, params = [None, None, None, None], dparams = [None, None, None]):
        super(RNN_spike, self).__init__()
        
        # is_bias=False
        self.dense_1 = spike_dense(40*3,n,
                                    tauAdp_inital_std=50,tauAdp_inital=150,
                                    tauM = 20,tauM_inital_std=5,device=device,bias=is_bias)
        self.rnn_1 = spike_rnn(n,n,tauAdp_inital_std=50,tauAdp_inital=150,
                                    tauM = 20,tauM_inital_std=5,device=device,bias=is_bias)
        self.dense_2 = readout_integrator(n,12,tauM=10,tauM_inital_std=1,device=device,bias=is_bias)

        self.thr = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.thr,5e-2)

        torch.nn.init.kaiming_normal_(self.rnn_1.recurrent.weight)
        
        torch.nn.init.xavier_normal_(self.dense_1.dense.weight)
        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)
      
        if is_bias:
            torch.nn.init.constant_(self.rnn_1.recurrent.bias,0)
            torch.nn.init.constant_(self.dense_1.dense.bias,0)
            torch.nn.init.constant_(self.dense_2.dense.bias,0)
        self.params = params
        self.dparams = dparams
    def forward(self,input):

        b,channel,seq_length,input_dim = input.shape
        hidden_spike_ = []
        hidden_spike2_ = []
        output = 0
        self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.rnn_1.set_neuron_state(b)

        self.dense_1.quantize(self.params[1])
        self.rnn_1.quantize(self.params[2])
        self.dense_2.quantize(self.params[3])
        hidden_mem = [0, 0, 0]
        b_mem = [0, 0]
        self.dense_1.dquantize(self.dparams[0])
        self.rnn_1.dquantize(self.dparams[1])
        self.dense_2.dquantize(self.dparams[2])
        if self.params[0] is None:
            input_s = input
            # print(torch.maximum(torch.max(input_s), torch.abs(torch.min(input_s))))
            # 11.1
            #input_s = thr_func(input-self.thr)*1.-thr_func(-self.thr-input)*1.
        else:
            if(self.params[0][1] == 1):
                i_pos = 1
            else:
                i_pos = self.params[0][1] - 1
            input_s = torch.fake_quantize_per_tensor_affine(input, self.params[0][0], 0, -self.params[0][1], i_pos).to(device)
            # print()      
        spike_layer1 = spike_layer2 = torch.zeros(b,n).to(device)
        mem3 = torch.zeros(b, 12).to(device)
        for i in range(seq_length):

            input_x = input_s[:,:,i,:].reshape(b,channel*input_dim)
            spike_layer1, mem_layer1, bh1  = self.dense_1.forward(input_x)
            spike_layer2, mem_layer2, bh2 = self.rnn_1.forward(spike_layer1)
            mem3 = self.dense_2.forward(spike_layer2)            
            output += mem3
            hidden_spike_.append(spike_layer1.data.cpu().numpy().mean())            
            hidden_spike2_.append(spike_layer2.data.cpu().numpy().mean()) 
            if(torch.max(torch.abs(mem_layer1)).item() > hidden_mem[0]):
                hidden_mem[0] = torch.max(torch.abs(mem_layer1)).item()
            if(torch.max(torch.abs(mem_layer2)).item() > hidden_mem[1]):
                hidden_mem[1] = torch.max(torch.abs(mem_layer2)).item()
            if(torch.max(torch.abs(mem3)).item() > hidden_mem[2]):
                hidden_mem[2] = torch.max(torch.abs(mem3)).item()  
            if(torch.max(bh1).item() > b_mem[0]):
                b_mem[0] = torch.max(bh1).item()
            if(torch.max(bh2).item() > b_mem[1]):
                b_mem[1] = torch.max(bh2).item()             
        output = F.log_softmax(output/seq_length,dim=1)
        #return output, torch.count_nonzero(input_s), torch.numel(input_s)
        return output, hidden_mem, b_mem
    def train(self, epochs, path='./model_baseline_hyperparam3/'):
        acc_list = []
        best_acc = 0
        best_val = 0
        learning_rate = 1e-2
        if is_bias:
            base_params = [
                            self.dense_1.dense.weight,
                            self.dense_1.dense.bias,
                            self.rnn_1.dense.weight,
                            self.rnn_1.dense.bias,
                            self.rnn_1.recurrent.weight,
                            self.rnn_1.recurrent.bias,
                            self.dense_2.dense.weight,
                            self.dense_2.dense.bias,
                            ]
        else:
            base_params = [
                            self.dense_1.dense.weight,
                            self.rnn_1.dense.weight,
                            self.rnn_1.recurrent.weight,
                            self.dense_2.dense.weight,
                            ]
        optimizer = torch.optim.Adam([
                                    {'params': base_params, 'lr': learning_rate},
                                    # {'params': self.thr, 'lr': learning_rate*0.01},
                                    {'params': self.dense_1.tau_m, 'lr': learning_rate * 2},  
                                    {'params': self.dense_2.tau_m, 'lr': learning_rate * 2},  
                                    {'params': self.rnn_1.tau_m, 'lr': learning_rate * 2},  
                                    {'params': self.dense_1.tau_adp, 'lr': learning_rate * 2.},  
                                    #{'params': self.dense_2.tau_adp, 'lr': learning_rate * 10},  
                                    {'params': self.rnn_1.tau_adp, 'lr': learning_rate * 2.}
                                    #{'params': self.rnn_1.tau_adp_param, 'lr': learning_rate * 2}, 
                                    #{'params': self.dense_1.tau_adp_param, 'lr': learning_rate * 2}

                                    ],
                                lr=learning_rate)
        criterion = nn.CrossEntropyLoss()#nn.NLLLoss()
        scheduler = StepLR(optimizer, step_size=20, gamma=.75) # 20
        for epoch in range(epochs):
            train_acc = 0
            sum_sample = 0
            train_loss_sum = 0
            for i, (images, labels) in enumerate(train_dataloader):
                images = images.view(-1,3,101, 40).to(device)
                labels = labels.view((-1)).long().to(device)

                optimizer.zero_grad()
                predictions = self.forward(images)
                _, predicted = torch.max(predictions.data, 1)

                train_loss = criterion(predictions,labels)
                train_loss.backward()
                train_loss_sum += train_loss.item()
                optimizer.step()

                labels = labels.cpu()
                predicted = predicted.cpu().t()
                train_acc += (predicted ==labels).sum() 
                sum_sample+=predicted.numel()

            if scheduler:
                scheduler.step()
            train_acc = train_acc.data.cpu().numpy()/sum_sample
            valid_acc = self.test(valid_dataloader,1)
            ts_acc = self.test(test_dataloader, 1)
            train_loss_sum+= train_loss
            acc_list.append(train_acc)
            print('lr: ',optimizer.param_groups[0]["lr"])
            if valid_acc>0.88 and train_acc>0.90:
                if ts_acc > best_acc:
                    best_acc = ts_acc
                    torch.save(self, path+str(ts_acc)+'-srnn-gsc.pth')
                elif ts_acc > 0.9:
                    torch.save(self, path+str(ts_acc)+'-srnn-gsc.pth')
                elif valid_acc > best_val or valid_acc > 0.89:
                    best_val = valid_acc
                    torch.save(self, path+str(ts_acc)+'v-srnn-gsc.pth')
            print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Valid Acc: {:.4f}'.format(epoch,
                                                                            train_loss_sum/len(train_dataloader),
                                                                            train_acc, ts_acc, valid_acc), flush=True)
        return acc_list

    def test(self, data_loader=test_dataloader, is_show = 0):
        test_acc = 0.
        sum_sample = 0.
        max_input = 0
        irt = 0
        ist = 0     
        fr0 = []
        fr1 = []
        mem_ = [0, 0, 0]
        h_ = [0, 0]  
        for i, (images, labels) in enumerate(data_loader):
            images = images.view(-1,3,101, 40).to(device)
            labels = labels.view((-1)).long().to(device)
            max_i = torch.maximum(torch.max(images), torch.abs(torch.min(images))).item()
            if max_i > max_input:
                max_input = max_i
            predictions, mem, h = self.forward(images)
            # # irt += ir
            # # ist += isz
            # for fr in ir:
            #     fr0.append(fr)
            # for fr in isz:
            #     fr1.append(fr)
            for i in range (3):
                if mem_[i] < mem[i]:
                    mem_[i] = mem[i]
            for i in range(2):
                if h_[i] < h[i]:
                    h_[i] = h[i]
            _, predicted = torch.max(predictions.data, 1)
            labels = labels.cpu()
            predicted = predicted.cpu().t()
            test_acc += (predicted ==labels).sum() 
            sum_sample+=predicted.numel()
        # print(f"Mean FR h1: {mem_}")
        # print(f"Mean FR h2: {h_}") 
        # print(irt/ist)           
        return test_acc.data.cpu().numpy()/sum_sample, irt, ist

    def finetune(self, qat_epochs=25, path='./qspinn/', id='0', realtime = False, fixed_pt = False, params = None):
        acc_list = []
        best_acc = 0
        best_val = 0
        ts_acc, _, _ = self.test(test_dataloader)
        model_return = copy.deepcopy(self.state_dict())
        best_acc = ts_acc
    
        learning_rate = 1e-2

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,eps=1e-5)
        criterion = nn.CrossEntropyLoss()#nn.NLLLoss()
        scheduler = StepLR(optimizer, step_size=20, gamma=.75) # 20    

        for epoch in range(qat_epochs):
            train_acc = 0
            sum_sample = 0
            train_loss_sum = 0
            for i, (images, labels) in enumerate(train_dataloader):
                if realtime:
                    self.params = printModelRange(self, params = params, fixedpt=fixed_pt)

                images = images.view(-1,3,101, 40).to(device)
                labels = labels.view((-1)).long().to(device)
                optimizer.zero_grad()
                predictions, _, _ = self.forward(images)

                _, predicted = torch.max(predictions.data, 1)

                train_loss = criterion(predictions,labels)
                
                # print(predictions,predicted)
                train_loss.backward()
                train_loss_sum += train_loss.item()
                optimizer.step()

                labels = labels.cpu()
                predicted = predicted.cpu().t()

                train_acc += (predicted ==labels).sum() 
                sum_sample+=predicted.numel()

            if scheduler:
                scheduler.step()
            train_acc = train_acc.data.cpu().numpy()/sum_sample
            valid_acc, _, _ = self.test(valid_dataloader)
            ts_acc, _, _ = self.test(test_dataloader)
            train_loss_sum+= train_loss
            acc_list.append(train_acc)
            print('lr: ',optimizer.param_groups[0]["lr"])
            if valid_acc>0.85 and train_acc>0.90:
                if ts_acc > best_acc:
                    best_acc = ts_acc
                    torch.save(self, path+str(ts_acc)+'-srnn-gsc2-'+id+'F.pth')
                elif ts_acc > 0.9:
                    torch.save(self, path+str(ts_acc)+'-srnn-gsc2-'+id+'F.pth')
                elif valid_acc > best_val or valid_acc > 0.89:
                    best_val = valid_acc
                    torch.save(self, path+str(ts_acc)+'v-srnn-gsc2-'+id+'F.pth')
            print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Valid Acc: {:.4f}'.format(epoch,
                                                                            train_loss_sum/len(train_dataloader),
                                                                            train_acc, ts_acc, valid_acc), flush=True)
        #self.load_state_dict(model_return)
        return best_acc, model_return

def printModelRange (model, params = None, fixedpt = False, verbose = False):

        y = (1 - torch.exp(-1. * 1 / model.dense_2.tau_m)).to(device)
        x = torch.einsum('ji,j->ji', model.dense_2.dense.weight, y).clone().detach()
        maxw = torch.maximum(torch.max(x), torch.abs(torch.min(x))).item()
        if verbose:
            print('Weight out max: ',torch.max(x))
            print('Weight out min: ',torch.min(x))
        b = model.dense_2.dense.bias
        b = torch.einsum('j, j -> j', b, y).clone().detach()
        maxb = torch.maximum(torch.max(b), torch.abs(torch.min(b))).item()
        if verbose:
            print('Bias out max', torch.max(b))
            print('Bias out bias min', torch.min(b))
        # print('tau out min: ', torch.min(model.dense_2.tau_m))
        # print('tau out max: ', torch.max(model.dense_2.tau_m))
        # print('tau out mean: ', torch.mean(model.dense_2.tau_m))
        mparams = [None, None, None, None]
        if params is not None:
            mparams[3] = [0, 0, 0, 0]
            mparams[3][1] = 2**(params[1]-1)
            mparams[3][3] = 2**(params[1]-1)
            mparams[0] = [0, 0]
            mparams[0][1] = 2**(params[0]-1)
            if fixedpt:
                if maxw < 0.001:
                    maxw = 0.001
                if maxb < 0.001:
                    maxb = 0.001 
                w_range = 2**math.ceil(math.log2(maxw))
                b_range = 2**math.ceil(math.log2(maxb))
                i_range = 16
            else:
                w_range = maxw
                b_range = maxb
                i_range = 11.1
            mparams[3][0] = w_range/(2**(params[1]-1))
            mparams[3][2] = b_range/(2**(params[1]-1))
            mparams[0][0] = i_range/(2**(params[0]-1))

        y = (1 - torch.exp(-1. * 1 / model.dense_1.tau_m)).to(device)
        x = torch.einsum('ji,j->ji', model.dense_1.dense.weight, y).clone().detach()
        maxw = torch.maximum(torch.max(x), torch.abs(torch.min(x))).item()
        b1 = model.dense_1.dense.bias.clone().detach()
        b1 = torch.einsum('j, j -> j', b1, y).clone().detach()        
        maxb = torch.maximum(torch.max(b1), torch.abs(torch.min(b1))).item()
        
        # w1 = model.dense_1.dense.weight.clone().detach()
        #w1[214, :] = 0
        if verbose:
            print('Weight in max: ',torch.max(x))
            print('Weight in min: ',torch.min(x))

        if params is not None:
            mparams[1] = [0, 0, 0, 0]
            mparams[1][1] = 2**(params[0]-1)
            mparams[1][3] = 2**(params[0]-1)

            if fixedpt:
                if maxw < 0.001:
                    maxw = 0.001
                if maxb < 0.001:
                    maxb = 0.001    
                # print(maxw)
                w_range = 2**math.ceil(math.log2(maxw))
                b_range = 2**math.ceil(math.log2(maxb))
            else:
                w_range = maxw
                b_range = maxb
            mparams[1][0] = w_range/(2**(params[0]-1))
            mparams[1][2] = b_range/(2**(params[0]-1))

        # #b1 [214] = 0
        if verbose:
            print('Bias max: ', torch.max(b1))
            print('Bias min: ', torch.min(b1))
        # a1 = torch.exp(-1. * 1 / model.dense_1.tau_m).clone().detach()
        # print('Alpha in mean: ', sum(a1)/len(a1))

        # r1 = torch.exp(-1. * 1 / model.dense_1.tau_adp).clone().detach()

        # print('Rho in mean: ', sum(r1)/len(r1))

        # print('Alpha1 max: ', torch.max(a1))
        # print('Alpha1 min: ', torch.min(a1))
        # print('Ro1 max: ', torch.max(r1))
        # print('Ro1 min: ', torch.min(r1))

        y = (1 - torch.exp(-1. * 1 / model.rnn_1.tau_m)).to(device)
        x = torch.einsum('ji,j->ji', model.rnn_1.dense.weight, y)
        maxw0 = torch.maximum(torch.max(x), torch.abs(torch.min(x))).item()
        if verbose:
            print('Weight2F max: ',torch.max(x))
            print('Weight2F min: ',torch.min(x))
        x = torch.einsum('ji,j->ji', model.rnn_1.recurrent.weight, y)
        maxw1 = torch.maximum(torch.max(x), torch.abs(torch.min(x))).item()
        if maxw0 > maxw1:
            maxw = maxw0
        else:
            maxw = maxw1

        b2 = model.rnn_1.recurrent.bias + model.rnn_1.dense.bias
        b2 = torch.einsum('j, j -> j', b2, y).clone().detach()        
        maxb = torch.maximum(torch.max(b2), torch.abs(torch.min(b2))).item()
        if verbose:
            print('Weight2R max: ',torch.max(x))
            print('Weight2R min: ',torch.min(x))
            print('bias max:', torch.max(b2))
            print('bias min:', torch.min(b2))
        if params is not None:
            mparams[2] = [0, 0, 0, 0]
            mparams[2][1] = 2**(params[0]-1)
            mparams[2][3] = 2**(params[0]-1)

            if fixedpt:
                if maxw < 0.001:
                    maxw = 0.001
                if maxb < 0.001:
                    maxb = 0.001                
                w_range = 2**math.ceil(math.log2(maxw))
                b_range = 2**math.ceil(math.log2(maxb))
            else:
                w_range = maxw
                b_range = maxb
            mparams[2][0] = w_range/(2**(params[0]-1))
            mparams[2][2] = b_range/(2**(params[0]-1))  
        if verbose:
           print(mparams[0][0]*mparams[0][1])
           print(mparams[1][0] * mparams[1][1], mparams[1][3]*mparams[1][2])
           print(mparams[2][0] * mparams[2][1], mparams[2][3]*mparams[2][2])
           print(mparams[3][0] * mparams[3][1], mparams[3][3]*mparams[3][2])
        return mparams
        #b2 = torch.einsum('j, j -> j', b2, y).clone().detach()        
        # print('Bias2 max: ', torch.max(b2))
        # print('Bias2 min: ', torch.min(b2))
        # print('Alpha2 mean: ', torch.mean(torch.exp(-1. * 1 / model.rnn_1.tau_m)))
        # print('Ro2 mean: ', torch.mean(torch.exp(-1. * 1 / model.rnn_1.tau_adp)))
        # print('Alpha2 max: ', torch.max(torch.exp(-1. * 1 / model.rnn_1.tau_m)))
        # print('Alpha2 min: ', torch.min(torch.exp(-1. * 1 / model.rnn_1.tau_m)))
        # print('Ro2 max: ', torch.max(torch.exp(-1. * 1 / model.rnn_1.tau_adp)))
        # print('Ro2 min: ', torch.min(torch.exp(-1. * 1 / model.rnn_1.tau_adp)))

        





def save_model(model):

    #bias1 = model.dense_1.dense.bias.to("cpu").detach().numpy()
    bias1 = model.dense_1.bias_q.to("cpu").detach().numpy()
    pd.DataFrame(bias1).to_csv("bias1.csv")    
    #bias2 = (model.rnn_1.dense.bias + model.rnn_1.recurrent.bias).to("cpu").detach().numpy()
    bias2 = model.rnn_1.bias_q.to("cpu").detach().numpy()
    pd.DataFrame(bias2).to_csv("bias2.csv")
    bias3 = model.dense_2.dense.bias
    bias3 = torch.einsum('j,j->j', bias3, 1. - torch.exp(-1 * 1 / model.dense_2.tau_m)).to("cpu").detach().numpy()
    pd.DataFrame(bias3).to_csv("bias3.csv")
    
    alpha1 = torch.exp(-1 * 1 / model.dense_1.tau_m).to("cpu").detach().numpy()
    pd.DataFrame(alpha1).to_csv("alpha1.csv")    
    alpha2 = torch.exp(-1 * 1 / model.rnn_1.tau_m).to("cpu").detach().numpy()
    pd.DataFrame(alpha2).to_csv("alpha2.csv") 
    alpha3 = torch.exp(-1 * 1 / model.dense_2.tau_m).to("cpu").detach().numpy()
    pd.DataFrame(alpha3).to_csv("alpha3.csv") 
    wi2h1 = model.dense_1.weight_q
    wi2h1 = torch.einsum('ji,j->ji', wi2h1, 1. -torch.exp(-1 * 1 / model.dense_1.tau_m)).to("cpu").detach().numpy()
    pd.DataFrame(wi2h1).to_csv("wi2h1.csv")
    wh1h2 = model.rnn_1.dense_q
    wh1h2 = torch.einsum('ji,j->ji', wh1h2, 1. - torch.exp(-1 * 1 / model.rnn_1.tau_m)).to("cpu").detach().numpy()
    pd.DataFrame(wh1h2).to_csv("wh1h2.csv")
    wh2h2 = model.rnn_1.recurrent_q
    wh2h2 = torch.einsum('ji,j->ji', wh2h2, 1. - torch.exp(-1 * 1 / model.rnn_1.tau_m)).to("cpu").detach().numpy()
    pd.DataFrame(wh2h2).to_csv("wh2h2.csv")
    wh2ho = model.dense_2.dense.weight
    wh2ho = torch.einsum('ji,j->ji', wh2ho, 1. - torch.exp(-1 * 1 / model.dense_2.tau_m)).to("cpu").detach().numpy()
    pd.DataFrame(wh2ho).to_csv("wh2o.csv")


learning_rate = 1e-2#3e-3#1e-2#3e-3#3e-4#1.2e-2
model = RNN_spike()
model2 = torch.load('./model_90.6-GSC-Baseline.pth')
model.load_state_dict(model2.state_dict())
model.to(device)
print('Baseline fp32 accuracy:')
print(model.test())
model.params = [[0.03125, 128], [0.015625, 128, 0.015625, 128], [0.015625, 128, 0.015625, 128], [0.25, 1, 0.25, 1]] #8bit params
print('int8 params:', model.params)
print('Baseline int8 accuracy:')
print(model.test())



model = torch.load('./model_90.9-GSC-Ternary.pth')
model.to(device)
print('Ternary params:', model.params)
print('Ternary model accuracy:')
print(model.test())


#Example usage of QMTS
#from qmts import *
#QMTS Step 1 example:
# flag = [True, True, True]
# acc_limit = 0.897
# # print(acc_limit)
# i = 1
# while(any(flag)):
#     if i != 0:
#         flag[i] = PTQ/QAT_combostep(model, i, acc_limit)
#     else:
#         flag[i] = PTQ/QAT_step(model, i, 1, acc_limit)
#     i = (i+1)%3

#Ternary Step
#T_step(model, 3, acc_limit, True)
# T_step(model, 0, acc_limit, False)
# T_step(model, 1, acc_limit, True)
# T_step(model, 2, acc_limit, True)

#QMTS Step 2 example:
#acc_limit=0.94
# scale_calibrate(model, 0, 2, acc_limit)
# scale_calibrate(model, 0, 5, acc_limit)
# scale_calibrate(model, 1, 2, acc_limit)
# scale_calibrate(model, 1, 5, acc_limit)
# scale_calibrate(model, 2, 2, acc_limit)

# while(any(flag)):
#     flag[0] = range_constrain(model, 0, 2, acc_limit)
#     flag[1] = range_constrain(model, 0, 5, acc_limit)
#     flag[2] = range_constrain(model, 1, 2, acc_limit)
#     flag[3] = range_constrain(model, 1, 5, acc_limit)
#     flag[4] = range_constrain(model, 2, 2, acc_limit)