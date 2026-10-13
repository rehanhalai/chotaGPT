import torch

with open("shakespear.txt","r") as file:
    text = file.read()

chars = sorted(list(set(text)))

encoder = {ch:i for i,ch in enumerate(chars)}
decoder = {i:ch for i,ch in enumerate(chars)}

data = torch.tensor([encoder[c] for c in text],dtype=torch.long)
print(data[:100])
