import torch
import torch.nn as nn

import numpy as np
import sys
sys.path.append("../../")
import adder


# Network
class SingleAdder(nn.Module):
    def __init__(self):
        super(SingleAdder, self).__init__()
        self.adder1 = adder.adder2d(5, 5, kernel_size=3, stride=1, padding=0, bias=False)
        self.adder1.adder = torch.nn.Parameter(torch.nn.init.ones_(torch.empty(5, 5, 3, 3)))

    def forward(self, x):
        return self.adder1(x)

    def get_weights(self):
        return self.state_dict()


def print_feature_map(feature_array, c):
    for i in range(c):
        print(feature_array[0, i])


def main():
    adder_model = SingleAdder()
    weights = adder_model.get_weights()
    print(weights['adder1.adder'].numpy())

    input = torch.empty(1, 5, 5, 5)
    input.fill_(1)
    print_feature_map(input.numpy(), 5)
    output = adder_model.forward(input)
    print_feature_map(output.detach().numpy(), 5)


if __name__ == '__main__':
    main()
