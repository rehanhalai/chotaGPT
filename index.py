
with open("shakespear.txt","r") as file:
    text = file.read()

chars = sorted(list(set(text)))

encoder = {ch:i for i,ch in enumerate(chars)}
decoder = {i:ch for i,ch in enumerate(chars)}

test = [encoder[c] for c in "Hello World"]
decoded_test = [decoder[i] for i in test]

print(test)
print(''.join(decoded_test))

