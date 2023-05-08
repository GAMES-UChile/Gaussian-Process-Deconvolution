import numpy as np
import torch

def print_loss(loss_dict, step):
    names = loss_dict.keys()
    if step % 30 == 0:
        for name in names:
            print(name.ljust(15), end="")
        print()
    for name in names:
        if name == 'inducing points':
            ind = loss_dict[name]
            print(f"{torch.min(ind):.1f} -- {torch.max(ind):.1f}".ljust(15), end="")
        else:
           print(f"{loss_dict[name]:.2f}".ljust(15), end="")
    print()

