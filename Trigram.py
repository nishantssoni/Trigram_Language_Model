import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split

class Trigram:
    def __init__(self,data,generator_seed = None):
        self.data = data
        self.d_stoi = {}
        self.s_stoi = {}
        self.d_itos = {}
        self.s_itos = {}
        self.xs = []
        self.ys = []
        self.seed = generator_seed

        g = torch.Generator().manual_seed(self.seed)
        self.W = torch.randn((702,27), generator=g, requires_grad=True)

        # preprocess
        self.preprocess()
        
    
    def preprocess(self):
        # # preprocessing
        a = '.abcdefghijklmnopqrstuvwxyz'
        count = 0
        for i in a:
            for j in a[1:]:
                self.d_stoi[(i+j)] = count
                count += 1
        
        count = 0
        for i in a:
            self.s_stoi[i] = count
            count += 1
        
        # reverse stoi
        self.d_itos = {i:s for s,i in self.d_stoi.items()}
        self.s_itos = {i:s for s,i in self.s_stoi.items()}
    
    def __repr__(self):
        pass
    
    def train(self,no_ite, step_size, regularization_loss):     
        # my code
        for w in self.data:
            w = '.' + w + '.'
            length = len(w)
            if length > 1:
                for i in range(length - 1): 
                    try: 
                        ix1 = self.d_stoi[(w[i:i+2])]
                        ix2 = self.s_stoi[(w[i+2])]
                    except:
                        continue
                    self.xs.append(ix1)
                    self.ys.append(ix2)
                    
        self.xs = torch.tensor(self.xs)
        self.ys = torch.tensor(self.ys)
        num = self.xs.nelement()

        # gradient decent
        xenc = F.one_hot(self.xs, num_classes=702).float() #cast to float becaue we feed float in NN
        for k in range(no_ite):
            # forward pass
            
            logits = xenc @ self.W #log-counts
            counts = logits.exp() #eqivalent to N from above
            prob = counts / counts.sum(1, keepdim=True)
            loss = -prob[torch.arange(num), self.ys].log().mean() + regularization_loss*(self.W**2).mean()
            print(loss.item())
        
            # backward pass
            self.W.grad = None #set grad to zero gradient
            loss.backward()
        
            # update
            self.W.data += -(step_size) * self.W.grad
        

        
    def sample(self, num, max_length=10):
        g = torch.Generator().manual_seed(self.seed)
        out = []
        for i in range(num):
            name =''
            is_start = True
            while True:
                if is_start:
                    ix = torch.randint(26,(1,),generator=g).item()
                    name +=self.d_itos[ix]
                    is_start = False
                else:
                    to_fed = name[-2:] # last two string
                    ix = self.d_stoi[to_fed]
                
                xenc = F.one_hot(torch.tensor([ix]), num_classes=702).float()
                logits = xenc @ self.W # predict log-counts
                counts = logits.exp() # counts, equivalent to N
                p = counts / counts.sum(1, keepdims=True)
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                name += self.s_itos[ix]
                
                if ix == 0 or len(name) > max_length:  #if its find '.'
                    if len(name) > 15: name += '.'
                    break
        
            out.append(name)
        return out
    
    def test(self, data):
        xs = []
        ys = []
        for w in data:
            w = '.' + w + '.'
            length = len(w)
            if length > 1:
                for i in range(length - 1): 
                    try: 
                        ix1 = self.d_stoi[(w[i:i+2])]
                        ix2 = self.s_stoi[(w[i+2])]
                    except:
                        continue
                    xs.append(ix1)
                    ys.append(ix2)
                    
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        num = xs.nelement()

        # gradient decent
        xenc = F.one_hot(xs, num_classes=702).float() #cast to float becaue we feed float in NN
        
        # forward pass 
        logits = xenc @ self.W #log-counts
        counts = logits.exp() #eqivalent to N from above
        prob = counts / counts.sum(1, keepdim=True)
        loss = -prob[torch.arange(num), ys].log().mean()
        return loss.item()   


if __name__=="__main__":
    # data loading
    words = open("names.txt", 'r').read().splitlines()
    
    # train 90% and test 10%
    train_set, test_set = torch.utils.data.random_split(words, [0.9, 0.1])
    train_set = list(train_set)
    test_set = list(test_set)

    # class init
    tri = Trigram(train_set,23934579)
    # training
    tri.train(no_ite=500, step_size=50.0, regularization_loss=0.01)

    # samples from model
    samples = tri.sample(5,max_length=8)
    print("\nsample from model :: ",samples)

    # model testing
    loss = tri.test(test_set)
    print("\navg negative log likelihood :: ",loss)
