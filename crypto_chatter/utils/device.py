import torch

if torch.cuda.is_available():
    print("we are using cuda 🏎🏎🏎")
    device = "cuda"
else:
    print("we are using cpu 🐌🐌🐌")
    device = "cpu"
