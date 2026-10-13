import torch

with open("shakespear.txt","r") as file:
    text = file.read()
chars = sorted(list(set(text)))

# character/string -> int or vice versa
encoder = {ch:i for i,ch in enumerate(chars)}
decoder = {i:ch for i,ch in enumerate(chars)}

full_data = torch.tensor([encoder[c] for c in text],dtype=torch.long)

# splitting data in train and validation set (90% train, 10% validation)
n = int(0.9 * len(full_data))
train_data = full_data[:n]
val_data = full_data[n:]

block_size = 4
batch_size = 8

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[ix+i] for i in range(block_size)])
    y = torch.stack([data[ix+i+1] for i in range(block_size)])
    return x , y

xb , yb = get_batch('train')

print(xb)
print(yb)