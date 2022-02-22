import numpy as np

def print_loss(loss_dict, step):
    names = loss_dict.keys()
    if step % 30 == 0:
        for name in names:
            print(name.ljust(15), end="")
        print()
    for name in names:
        print(f"{loss_dict[name]:.2f}".ljust(15), end="")
    print()

