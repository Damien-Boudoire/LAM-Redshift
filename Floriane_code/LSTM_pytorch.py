from multiprocessing import cpu_count
from pathlib import Path
from dataset_redshift import dataset_redshift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

seed = 1
np.random.seed(seed)
#torch.cuda.set_device(0)  # if you have more than one CUDA device

print('hello')
X_train, X_valid ,X_test, target_train, target_valid ,target_test , Y_train , Y_validation, Y_test , nb_class, nom_classes = dataset_redshift('Flag_class',Undersample=True)

X_valid=np.reshape(X_valid, (X_valid.shape[0], 1,X_valid.shape[1]))
X_train=np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))  

X_train, X_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid)]
y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (target_train, target_valid)]


print(X_train.shape)
#enc = LabelEncoder()
#y_enc = enc.fit_transform(y)
#print()
train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_valid, y_valid)

bs = 32 

train_dl = DataLoader(train_ds, bs, shuffle=True)
valid_dl = DataLoader(valid_ds, bs, shuffle=False)

def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()

class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]

class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act=nn.Sigmoid()
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.act(self.fc(out[:, -1, :]))
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        #print(np.shape(h0))
        return [t for t in (h0, c0)]

input_dim = 17908    
hidden_dim = 256
layer_dim = 1
output_dim = 3
seq_dim = bs

lr = 0.001
n_epochs = 100
iterations_per_epoch = len(train_dl)
best_acc = 0
patience, trials = 100, 0

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
#model = model.cuda()
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)
#sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/100))

print('Start model training')

for epoch in range(1, n_epochs + 1):
    
    for i, (x_batch, y_batch) in enumerate(train_dl):
        model.train()
        #x_batch = x_batch.cuda()
        #y_batch = y_batch.cuda()
        #sched.step()
        opt.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        opt.step()
    
    model.eval()
    correct, total = 0, 0
    for x_val, y_val in valid_dl:
        x_val, y_val = [t for t in (x_val, y_val)]
        out = model(x_val)
        #print(out)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        #print(preds)
        total += y_val.size(0)
        correct += (preds == y_val).sum().item()
    
    acc = correct / total

    #if epoch % 5 == 0:
    print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

    # if acc > best_acc:
    #     trials = 0
    #     best_acc = acc
    #     torch.save(model.state_dict(), 'best.pth')
    #     print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
    # else:
    #     trials += 1
    #     if trials >= patience:
    #         print(f'Early stopping on epoch {epoch}')
    #     break