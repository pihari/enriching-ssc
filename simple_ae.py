import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

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
        self.enc_l1 = nn.Linear(768, 768)
        self.enc_l2 = nn.Linear(768, 768)

        self.intermed = nn.Linear(768, 768)

        self.dec_l1 = nn.Linear(768, 768)
        self.dec_l2 = nn.Linear(768, 768)

        self.out = nn.Linear(768, 768)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.BCELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        enc1_out = F.relu(self.enc_l1(input))
        enc2_out = F.relu(self.enc_l2(enc1_out))

        interm = self.intermed(enc2_out)

        dec1_out = F.relu(self.dec_l1(interm))
        dec2_out = F.relu(self.dec_l2(dec1_out))

        out = torch.sigmoid(self.out(dec2_out))

        return out

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
            loss = self.SimpleAE.loss(pred, self.cur_target)
            print(f'batch: {str(batch+1)}, loss: {str(loss.item())}')
            loss.backward()
            self.next_batch()
            self.SimpleAE.optimizer.step()
        return self.SimpleAE

import matplotlib.pyplot as plt
import numpy as np

class AEVisualizer:
    def __init__(self, target, model):
        self.target = target
        self.model = model
        self.samples = self.select_random()
    
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

class Bertifier:
    def __init__(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        self.text_model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
        self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")

import os
if __name__ == '__main__':
    #data_dir = os.path.join("/data/s1/haritz", "emb_samples_np_0.csv")
    lr = 1e-5
    epochs = 20
    batch_size = 256
    ae = SimpleAE(lr)
    data_dir = "samples.csv"
    data = AEData(data_dir)
    samples = data.load_samples()
    #target, input = data.import_data()

    n_samp = len(samples)
    ae_init = False
    model = None
    for epoch in range(epochs):
        counter = 0
        t_list = []
        for index, row in samples.iterrows():
            print(f"Sample {counter} of {n_samp}.", end="\r")
            counter += 1
            ti = row['paper_tokens']
            ci = row['code_tokens']
            try:
                ti_tokens = text_tokenizer(ti, return_tensors="pt", padding=True)
                ti_emb = text_model(**ti_tokens, output_hidden_states=True)
                ti_emb_allhidden = ti_emb[2]
                ti_emb_avg = torch.mean(ti_emb_allhidden, dim=0) #axis=0?
                t_list.append(ti_emb_avg)
            except:
                break
            if counter % batch_size == 0:
                t_shape = list(t_list[0].size())
                t_shape[0] *= batch_size
                target = torch.Tensor(t_shape)
                input = torch.Tensor(t_shape)
                torch.cat(t_list, out=target)
                torch.cat(t_list, out=input)
                target = target.to(ae.device)
                input = input.to(ae.device)
                print(target.shape, input.shape)
                t_list = []

                if not ae_init:
                    learner = SimpleLeaner(ae, input, target, batch_size=batch_size, epochs=epochs)
                    model = learner.learn()
                    ae_init = True
                else:
                    learner.next_input(input, target)
                    model = learner.learn()

        visualizer = AEVisualizer(target, model)
        visualizer.visualize()