

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.EMB_SIZE = 768
        self.STATE_SIZE = 128
        self.y = torch.Tensor(1) #pseudo label (always 1)
        self.loss_data = []
    
    def forward(self, input, target, state):
        split_in = torch.split(input, self.EMB_SIZE, 1)
        split_tar = torch.split(target, self.EMB_SIZE, 1)
        split_state = torch.split(state, self.STATE_SIZE, 1)
        loss_p = F.mse_loss(split_in[0], split_tar[0])
        loss_c = F.mse_loss(split_in[1], split_tar[1])
        sim_enc = F.cosine_embedding_loss(split_state[0], split_state[1], self.y, reduction='none')
        loss_ctop = F.mse_loss(split_in[1], split_tar[0])
        cossim = nn.CosineSimilarity()
        loss_cos_sim_enc = 1 - cossim(split_state[0], split_state[1]).mean()
        #sim_enc = cos_sim_enc.mean()
        print(loss_p.item(), loss_c.item(), loss_cos_sim_enc.item())
        loss_ae = loss_p + loss_c + loss_cos_sim_enc
        self.loss_data.append((loss_p.item(), loss_c.item(), loss_cos_sim_enc.item()))
        return loss_ae

    def get_loss_data(self):
        return self.loss_data
    

class AEData:
    def __init__(self, dir):
        self.dir = dir

    def load_samples(self):
        return pd.read_csv(self.dir)

    def import_data(self):
        df = pd.read_csv(self.dir)
        target_data = df.iloc[:, 1:]
        input_data = df.iloc[:, 1:]
        target_data = torch.Tensor(target_data.values)
        input_data = torch.Tensor(input_data.values)
        return target_data, input_data

class SimpleAE(nn.Module):
    def __init__(self, lr=1e-5):
        super(SimpleAE, self).__init__()
        self.enc_l1 = nn.Linear(768, 512)
        self.enc_l2 = nn.Linear(512, 256)

        self.intermed = nn.Linear(256, 128)

        self.dec_l1 = nn.Linear(128, 256)
        self.dec_l2 = nn.Linear(256, 512)

        self.out = nn.Linear(512, 768)

        self.enc2_l1 = nn.Linear(768, 512)
        self.enc2_l2 = nn.Linear(512, 256)
        self.intermed2 = nn.Linear(256, 128)
        self.dec2_l1 = nn.Linear(128, 256)
        self.dec2_l2 = nn.Linear(256, 512)
        self.out2 = nn.Linear(512, 768)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = CustomLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.state = None

    def forward(self, input):
        EMB_SIZE = 768
        split_t = torch.split(input, EMB_SIZE, 1)
        input_paper = split_t[0]
        input_code = split_t[1]

        enc1_out = F.relu(self.enc_l1(input_paper))
        enc2_out = F.relu(self.enc_l2(enc1_out))

        interm = self.intermed(enc2_out)

        dec1_out = F.relu(self.dec_l1(interm))
        dec2_out = F.relu(self.dec_l2(dec1_out))

        paper_out = torch.sigmoid(self.out(dec2_out))

        enc1_out = F.relu(self.enc2_l1(input_code))
        enc2_out = F.relu(self.enc2_l2(enc1_out))

        interm2 = self.intermed2(enc2_out)

        dec1_out = F.relu(self.dec2_l1(interm2))
        dec2_out = F.relu(self.dec2_l2(dec1_out))

        code_out = torch.sigmoid(self.out2(dec2_out))

        self.state = torch.cat((interm, interm2), dim=1)

        return torch.cat((paper_out, code_out), dim=1)
    
    def get_state(self):
        return self.state
    
    def get_loss_data(self):
        return self.loss.get_loss_data()
    

from torch.autograd import Variable
class SimpleLeaner(object):
    def __init__(self, SimpleAE, input, target, batch_size=512, epochs=20):
        self.SimpleAE = SimpleAE
        self.input = Variable(input)
        self.input_dim = input.shape
        self.target = Variable(target)
        self.target_dim = target.shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.cur_batch = self.input[:self.batch_size,:]
        self.cur_target = self.target[:self.batch_size,:]
        self.batch_num = 0
        self.batch_num_total = self.input_dim[0] // self.batch_size + (self.input_dim[0] % self.batch_size != 0)
        self.loss_data = []
    
    def next_input(self, input, target):
        self.input = Variable(input)
        self.target = Variable(target)

    def reset(self):
        self.batch_num = 0
        self.cur_batch = self.input[:self.batch_size,:]
        self.cur_target = self.target[:self.batch_size,:]

    def next_batch(self):
        self.batch_num += 1
        self.cur_batch = self.input[self.batch_size * self.batch_num : self.batch_size * (self.batch_num+1),:]
        self.cur_target = self.target[self.batch_size * self.batch_num : self.batch_size * (self.batch_num+1),:]
        if self.batch_num == self.batch_num_total:
            self.reset()
    
    def learn(self):
        for batch in range(self.batch_num_total):
            self.SimpleAE.optimizer.zero_grad()
            pred = self.SimpleAE.forward(self.cur_batch)
            state = self.SimpleAE.get_state()
            loss = self.SimpleAE.loss(pred, self.cur_target, state)
            print(f'batch: {str(self.batch_num+1)}, loss: {str(loss.item())}')
            loss.backward()
            self.next_batch()
            self.SimpleAE.optimizer.step()
        return self.SimpleAE
    
    def get_loss_data(self):
        return self.SimpleAE.get_loss_data()

import matplotlib.pyplot as plt
import numpy as np

class DataVisualizer:
    def __init__(self, data):
        self.data = data
    
    def select_random(self):
        rnd = np.random.randint(0, self.target.shape[0]-1, 10)
        return rnd
    
    def visualize(self):
        for _, i in enumerate(self.samples):
            output = self.model.forward(self.target[i]).view(28, 28)
            cat_img = torch.cat((self.target[i].view(28, 28), output), 1)
            plt.imshow(cat_img.cpu().detach().numpy())
            plt.title('AE target / output')
            plt.savefig('./out_images/sample'+str(_+1))
            plt.close()
    
    def plot(self, i):
        loss_p_data = [d[0] for d in self.data]
        loss_c_data = [d[1] for d in self.data]
        loss_cse = [d[2] for d in self.data]
        loss_ae = [a+b+c for (a,b,c) in zip(loss_p_data, loss_c_data, loss_cse)]
        over_time = [i for i in range(len(loss_ae))]
        plt.figure()
        plt.plot(over_time, loss_p_data, label="reconstruction error paper")
        plt.plot(over_time, loss_c_data, label="reconstruction error code")
        plt.plot(over_time, loss_cse, label="cosine similarity loss")
        plt.plot(over_time, loss_ae, label="total loss")
        plt.autoscale()
        plt.legend()
        plt.savefig('./outimgs/loss_norm_2d')
        plt.close()
            


class Bertifier:
    def __init__(self):
        print("Initializing BERTs...")
        self.text_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        self.text_model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
        self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")

    def normalize_tensor(self, t):
        t_max = np.max(t)
        t_min = np.min(t)
        t = (t - t_min) / (t_max - t_min)
        return t
    
    def calc_tensor(self, data, text=True, is_2d=False):
        if text:
            tokenizer = self.text_tokenizer
            model = self.text_model
        else:
            tokenizer = self.code_tokenizer
            model = self.code_model
    
        tok = tokenizer(data, return_tensors="pt", padding=True, truncation=(not text or is_2d))
        emb = model(**tok, output_hidden_states=True)
        emb_allhidden = emb[2]
        emb_list = []
        for h in emb_allhidden:
            emb_list.append(h.detach().cpu().numpy())
        emb_np = np.asarray(emb_list)
        emb_avg = np.mean(emb_np, axis=0)
        emb_avg = self.normalize_tensor(emb_avg)
        #emb_avg += 1
        #emb_avg /= 2
        if not is_2d:
            emb_avg = np.mean(emb_avg, axis=1)
        else:
            emb_avg = emb_avg.flatten()
            print(np.shape(emb_avg))
        #print(emb_avg)
        return torch.from_numpy(emb_avg)

import os
from transformers import *
if __name__ == '__main__':
    #data_dir = os.path.join("/data/s1/haritz", "emb_samples_np_0.csv")
    lr = 1e-4
    epochs = 1
    batch_size = 512
    ae = SimpleAE(lr)
    data_dir = "samples.csv"
    data = AEData(data_dir)
    samples = data.load_samples()
    bert = Bertifier()
    #target, input = data.import_data()

    n_samp = len(samples)
    ae_init = False
    model = None
    is_2d = True
    for epoch in range(epochs):
        counter = 0
        t_list = []
        for index, row in samples.iterrows():
            print(f"Sample {counter} of {n_samp}.", end="\r")
            counter += 1
            ti = row['paper_tokens']
            ci = row['code_tokens']
            try:
                ti_tensor = bert.calc_tensor(ti, text=True, is_2d=is_2d)
                ci_tensor = bert.calc_tensor(ci, text=False, is_2d=is_2d)
                t_list.append((ti_tensor, ci_tensor))
            except Exception as e:
                print(e)
                break
            if counter % batch_size == 0:
                tp_list, tc_list = zip(*t_list)
                # print(tp_list, tc_list)
                t_shape = list(tp_list[0].size()) #prior pooling results in equal size
                t_shape[0] *= batch_size
                target_paper = torch.empty(t_shape)
                input_paper = torch.empty(t_shape)
                torch.cat(tp_list, out=target_paper)
                torch.cat(tp_list, out=input_paper)
                target_paper = target_paper.to(ae.device)
                input_paper = input_paper.to(ae.device)

                t_shape = list(tc_list[0].size()) #prior pooling results in equal size
                t_shape[0] *= batch_size
                target_code = torch.empty(t_shape)
                input_code = torch.empty(t_shape)
                torch.cat(tc_list, out=target_code)
                torch.cat(tc_list, out=input_code)
                target_code = target_code.to(ae.device)
                input_code = input_code.to(ae.device)

                input = torch.cat((input_paper, input_code), dim=1)
                target = torch.cat((target_paper, target_code), dim=1)
                #print(target.shape, input.shape)
                t_list = []

                if not ae_init:
                    learner = SimpleLeaner(ae, input, target, batch_size=batch_size, epochs=epochs)
                    model = learner.learn()
                    ae_init = True
                else:
                    learner.next_input(input, target)
                    model = learner.learn()

                plti = counter // batch_size
                loss_data = learner.get_loss_data()
                visualizer = DataVisualizer(loss_data)
                visualizer.plot(plti)
