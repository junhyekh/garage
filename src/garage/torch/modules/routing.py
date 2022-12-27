import torch
import torch.nn as nn
from torch.nn import functional as F


class Routing(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.context_dim = 7

        self.obs_condition = 'multiply'
        self.activation_func = 'relu'
        self.num_modules = 4
        self.num_layers = 4
        self.gating_input_dim = 128
        self.gating_output_dim = 128
        self.num_gating_layer = 3
        self.ctx_shape = 128

        self.gating_input_shape = self.ctx_shape
        if self.activation_func == 'relu':
            self.activation_func = F.relu

    
        self.upsample_ctx=torch.nn.Sequential(
            nn.Linear(7, self.gating_input_dim),
            nn.ReLU(),
            nn.Linear(self.gating_input_dim, self.gating_input_dim)
        )

        # Gating network
        self.gating_fcs = []
        for i in range(self.num_gating_layer):
            self.gating_fcs.append(
                nn.Linear(
                    self.gating_input_dim,
                    self.gating_output_dim))
        self.gating_fcs = nn.ModuleList(self.gating_fcs)

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []
        self.gating_weight_fc_0 = nn.Linear(
            self.gating_input_shape,
            self.num_modules * self.num_modules)

        for layer_idx in range(self.num_layers - 2):
            gating_weight_cond_fc = nn.Linear(
                (layer_idx + 1) * self.num_modules * self.num_modules,
                self.gating_input_shape)
            self.__setattr__(
                "gating_weight_cond_fc_{}".format(
                    layer_idx + 1),
                gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)
            gating_weight_fc = nn.Linear(
                self.gating_input_shape,
                self.num_modules * self.num_modules)
            self.__setattr__(
                "gating_weight_fc_{}".format(
                    layer_idx + 1),
                gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)
        self.gating_weight_fcs = nn.ModuleList(self.gating_weight_fcs)
        self.gating_weight_cond_fcs = nn.ModuleList(
            self.gating_weight_cond_fcs)

        self.gating_weight_cond_last = nn.Linear(
            (self.num_layers - 1) * self.num_modules * self.num_modules,
            self.gating_input_shape)
        self.gating_weight_last = nn.Linear(
            self.gating_input_shape, self.num_modules)

    def observation_condition(self, state, ctx):
        if self.obs_condition == 'multiply':
            ctx = ctx * state
            
        state = self.activation_func(state)
        ctx = self.activation_func(ctx)

        return state, ctx

    def gating_network(self, ctx):
        for i_fc in self.gating_fcs[:-1]:
            ctx = self.activation_func(i_fc(ctx))
        ctx = self.gating_fcs[-1](ctx)
        return ctx

    def one_stage(self, flatten_weights, gating_weight_fc,
                  gating_weight_cond_fc, ctx, weight_shape):
        cond = torch.cat(flatten_weights, dim=-1)
        # cond = self.activation_func(cond)
        cond = gating_weight_cond_fc(cond)
        # _, cond = self.observation_condition(ctx, cond)
        cond = cond * ctx
        cond = self.activation_func(cond)
        raw_weight = gating_weight_fc(cond).view(weight_shape)
        softmax_weight = F.softmax(raw_weight, dim=-1)
        return cond, softmax_weight

    def forward(self, state, ctx):
        # 1. condition observation
        ctx = self.upsample_ctx(ctx)
        ctx = ctx * state

        # 2. Gating network
        ctx = self.gating_network(ctx)

        # 3. Compute probabilities
        weight = []
        flatten_weights = []
        base_shape = ctx.shape[:-1]

        # 3- 1) First layer
        raw_weight = self.gating_weight_fc_0(self.activation_func(ctx))
        weight_shape = base_shape + torch.Size(
            [self.num_modules, self.num_modules])
        flatten_shape = base_shape + torch.Size(
            [self.num_modules * self.num_modules])
        raw_weight = raw_weight.view(weight_shape)
        softmax_weight = F.softmax(raw_weight, dim=-1)
        weight.append(softmax_weight)
        flatten_weights.append(softmax_weight.view(flatten_shape))

        # 3- 2) Middle layer
        for gating_weight_fc, gating_weight_cond_fc in zip(
                self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond, softmax_weight = self.one_stage(
                flatten_weights, gating_weight_fc, gating_weight_cond_fc, ctx, weight_shape)
            weight.append(softmax_weight)
            flatten_weights.append(softmax_weight.view(flatten_shape))

        # 3- 3) last layer
        cond = torch.cat(flatten_weights, dim=-1)
        cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * ctx
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim=-1)
        weight.append(last_weight)
        flatten_weights.append(last_weight.view(flatten_shape))
        return weight


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from pathlib import Path
    cfg = OmegaConf.load(Path('./cfg/policy.yaml').resolve())

    routing_exp = Routing(cfg)
    print(routing_exp(torch.rand(1, 128), torch.rand(1, 128)))

    # routing_weight = routing_exp.forward(obs, ctx)
