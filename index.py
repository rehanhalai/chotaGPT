
with open("shakespear.txt","r") as file:
    chars = sorted(list(set(file.read())))

print(''.join(chars))
print(len(chars))


