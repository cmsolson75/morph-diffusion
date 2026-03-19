import torch

def lerp(a, b, t):
    return (1 - t) * a + t * b

def lerp2(a, b, t):
    return a + t * (b - a)

if __name__ == "__main__":
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    t = 0.5

    breakpoint()