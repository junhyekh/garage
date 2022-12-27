
import numpy as np

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Distribution, Normal

from garage.torch.modules import BaseNetwork, Routing


class TanhNormal(Distribution):
    """
    Basically from RLKIT

    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                torch.zeros(self.normal_mean.size()),
                torch.ones(self.normal_std.size())
            ).sample().to(self.normal_mean.device)
        )

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def entropy(self):
        return self.normal.entropy()


class Policy(nn.Module):
    """
    Policy based on SAC.
    It takes input as state and context order.
    """

    def __init__(self):
        super().__init__()

        self.base=nn.Sequential(
            nn.Linear(39, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.network = BaseNetwork(8, 128)
        self.router = Routing()

    def forward(self, obs, ctx):
        state = self.base(obs)
        weights = self.router(state, ctx)
        # print('state', state.shape)
        # print('ctx', ctx.shape)
        # print('weights', [w.shape for w in weights])
        out = self.network(state, weights)
        # print('out', out.shape)
        mean, log_std = out.chunk(2, dim=-1)
        #log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        return mean, std, log_std

    def eval_act(self, state, ctx):
        with torch.no_grad():
           mean, std, log_std = self.forward(state, ctx)

        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()

    def get_distribution(self, state, ctx):

        mean, std, log_std = self.forward(state, ctx)
        dic = {}

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True)

        dic.update({
            "mean": mean,
            "log_std": log_std,
            "ent": ent
        })

        action, z = dis.rsample(return_pretanh_value=True)
        log_prob = dis.log_prob(
            action,
            pre_tanh_value=z
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        dic["log_prob"] = log_prob
        dic["action"] = action.squeeze(0)
        dic['pre_tanh_value'] = z
        return dic

    def get_action(self, state, ctx, deterministic:bool = False):        
        mean, std, log_std = self.forward(state, ctx)

        dis = TanhNormal(mean, std)

        action = dis.rsample(return_pretanh_value=False)
        if deterministic:
            return (action, mean)
        return action.cpu().numpy(), dict(mean=mean.cpu().numpy(), log_std=log_std.cpu().numpy())


class Qfunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=nn.Sequential(
            nn.Linear(43, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # FIXME(ycho): get this from cfg
        self.network = BaseNetwork(1, 128)
        self.router = Routing()

    def forward(self, obs, action, ctx):
        input = torch.cat([obs, action], dim=-1)
        input = self.base(input)
        weights = self.router(input, ctx)
        out = self.network(input, weights)

        return out


class StateEncoder(nn.Module):
    """
    State encoder that takes input as image, proprioceptive, and hidden state order.
    For image (B, T, C, H, W) and proprioceptive (x) (B, T, N)
    It outputs state (B, M) and hidden state (L, B, hidden shape)
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg["net"]["encoder"]["recurrent"]["hidden_size"]
        self.num_layer = cfg["net"]["encoder"]["recurrent"]["num_layers"]
        self.RNN_input_shape = cfg["net"]["encoder"]["recurrent"][
            "input_shape"]

        input_shape = cfg["net"]["encoder"]["input_shape"]
        latent_shape = cfg["net"]["encoder"]["latent_shape"]
        self.state_shape = cfg["net"]["encoder"]["state_shape"]

        linear = []
        for layer in cfg["net"]["encoder"]["proprioceptive_encoder"][
                "hidden_shapes"]:
            linear.append(nn.Linear(input_shape, layer))
            linear.append(nn.ReLU())
            input_shape = layer

        self.linear_encoder = nn.Sequential(*linear)
        image = []

        in_channel = 3
        for channel in cfg["net"]["encoder"]["image_encoder"]["channel"]:
            image.append(ImpalaBlock(in_channel, channel))
            in_channel = channel
        self.image_encoder = nn.Sequential(*image)
        self.featrue_dim_change = nn.Linear(latent_shape, self.RNN_input_shape)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layer)

        self.last_fc = nn.Linear(self.hidden_size, self.state_shape)

    def forward(self, image, obs, hidden):

        #embed input

        # x: (B, T, N) -> (B*T, N)
        
        if image is not None:
            batch, sequence, channel, height, width = image.shape
            image = image.view(batch * sequence, channel, height, width)
        else:
            batch, sequence, _ = obs.shape
        x = obs.view(batch * sequence, -1)
        linear_latent = self.linear_encoder(x)
        if image is not None:
            img_latent = self.image_encoder(
                image).view(linear_latent.shape[0], -1)

            # features : (B*T, M) -> (B,T,M) -> (T,B,M) -> (T,B,H_in)
            features = torch.cat([img_latent, linear_latent], -1)
        else:
            features = linear_latent
        features = F.relu(self.featrue_dim_change(features)).view(
            batch, sequence, -1).swapaxes(0, 1)

        # hidden should be : (2, 16, 256)
        # got              : (4, 16, 128)
        out, hidden = self.gru(features, hidden)

        out = out.view(batch * sequence, -1)

        # output layer
        output = self.last_fc(out)
        return output, hidden
    

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from pathlib import Path
    cfg = OmegaConf.load(Path('./cfg/policy.yaml').resolve())

    policy = Policy(cfg)
    Q = Qfunction(cfg)

    encoder = StateEncoder(cfg)

    #print(encoder(torch.rand(1,5,3,128,128), torch.rand(1,5,7), torch.zeros(2,1,256)))
    state, hidden = encoder(
        torch.rand(
            1, 5, 3, 128, 128), torch.rand(
            1, 5, 7), torch.zeros(
                2, 1, 256))

    ctx = torch.rand(1, 128)
    #Policy
    print(policy(state, ctx))
    print(policy.eval_act(state, ctx))
    print(policy.get_action(state, ctx))

    #Q
    print(Q(state, ctx))
