#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
#TODO:
# Distinguish image and proprioceptive input
# Initialization
# What is rigth fusion ?

"""


def init(module, weight_init, bias_init):
    weight_init(module.weight)
    bias_init(module.bias)


class BaseNetwork(nn.Module):
    """
    This network is the SM like network used in the policy and Q function for inference
    It takes state and weigths from routing network and output actions (mean, logstd)
    """

    def __init__(self, output_shape, input_shape) -> None:
        super().__init__()

        self.layers = []

        self.num_layers = 4
        self.num_modules = 4

        output_shape = output_shape

        module_input_shape = input_shape
        module_hidden_shape = 128
        for i in range(self.num_layers):
            layer_module = []
            for j in range(self.num_modules):
                module = nn.Linear(module_input_shape, module_hidden_shape)
                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i, j), module)

            module_input_shape = module_hidden_shape
            self.layers.append(layer_module)

        self.last = nn.Linear(module_input_shape, output_shape)

    def forward(self, state, routing_prob):

        module_outputs = [(layer_module(state))
                          for layer_module in self.layers[0]]

        module_outputs = torch.stack(module_outputs, -2)

        for i in range(1, self.num_layers - 1):
            new_module_outputs = []
            for j, module in enumerate(self.layers[i]):

                module_input = (module_outputs *
                                routing_prob[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = F.relu(module_input)
                new_module_outputs.append((
                    module(module_input)
                ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim=-2)
        out = (module_outputs * routing_prob[-1].unsqueeze(-1)).sum(-2)
        out = F.relu(out)
        out = self.last(out)

        return out
