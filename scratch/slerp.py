import torch
import torch.nn.functional as F

def slerp(a, b, t, eps=1e-6):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    dot = (a * b).sum()
    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)

    if torch.abs(theta) < eps:
        return (1 - t) * a + t * b
    
    sin_theta = torch.sin(theta)

    w1 = torch.sin((1 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta

    return w1 * a + w2 * b

if __name__ == "__main__":
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    t = 0.5

    x = slerp(a, b, t)

    print(x, x.norm())  # stays on unit circle

    breakpoint()