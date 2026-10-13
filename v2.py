import torch
import torch.nn as nn
from torch.nn import functional as F

with open("shakespear.txt","r") as file:
    text = file.read()
chars = sorted(list(set(text)))

# character/string -> int or vice versa
toi = {ch: i for i, ch in enumerate(chars)}
itc = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [toi[c] for c in s]
decode = lambda l: ''.join([itc[i] for i in l])

full_data = torch.tensor(encode(text),dtype=torch.long)

# splitting data in train and validation set (90% train, 10% validation)
n = int(0.9 * len(full_data))
train_data = full_data[:n]
val_data = full_data[n:]

block_size = 4
batch_size = 32
vocab_size = len(chars)
num_embedding = 32

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[ix+i] for i in range(block_size)])
    y = torch.stack([data[ix+i+1] for i in range(block_size)])
    return x , y

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embedding)
        self.positon_embedding_table = nn.Embedding(block_size,num_embedding)
        self.lm_head = nn.Linear(num_embedding,vocab_size)
    
    def forward(self, idx, target=None):
        
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.positon_embedding_table(idx)
        x = token_emb + pos_emb
        logits = self.lm_head(x)
    
        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)

        return logits, loss

    def generate(self,idx,max_token_count):
        # the current context is in (B,T) and we want to predict the next token
        # so we only want to focus on the last token in the sequence
        for _ in range(max_token_count):
            logits , loss = self(idx)
            logits = logits[:,-1,:] # (B,C)
            probs = F.softmax(logits,dim=-1) # (B,C)
            idx_next = torch.multinomial(probs,num_samples=1) # (B,1)
            idx = torch.cat((idx,idx_next),dim=1) # (B,T+1)
        return idx


model = BigramLanguageModel()

optimzer = torch.optim.AdamW(model.parameters(),lr=1e-3)

for i in range(10000):
    xb,yb = get_batch('train')
    logits , loss = model(xb,yb)
    optimzer.zero_grad(set_to_none=True)
    loss.backward()
    optimzer.step()

context = torch.zeros((1,1),dtype=torch.long)
print(decode(model.generate(idx= context,max_token_count=100)[0].tolist()))
