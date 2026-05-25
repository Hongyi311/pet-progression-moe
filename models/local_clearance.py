from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LocalOutputs:
    f_local: torch.Tensor
    gamma: torch.Tensor
    rho: torch.Tensor | None


class LocalClearance(nn.Module):
    def __init__(
        self,
        n_roi: int | None = None,
        hidden: int = 32,
        use_rho: bool = False,
        rho_max: float = 0.1,
        mode: str = "per_roi",
        global_hidden: int = 64,
        global_layers: int = 2,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.use_rho = use_rho
        self.rho_max = rho_max
        self.mode = str(mode).lower()
        if self.mode not in {"per_roi", "global"}:
            raise ValueError(f"Unknown local mode: {mode}")
        if self.mode == "global":
            if n_roi is None:
                raise ValueError("n_roi is required when local mode is 'global'.")
            in_dim = int(n_roi) + 1
            layers = [nn.Linear(in_dim, global_hidden), nn.ReLU()]
            for _ in range(max(global_layers - 2, 0)):
                layers.extend([nn.Linear(global_hidden, global_hidden), nn.ReLU()])
            layers.append(nn.Linear(global_hidden, int(n_roi)))
            self.gamma_net = nn.Sequential(*layers)
            if use_rho:
                rho_layers = [nn.Linear(in_dim, global_hidden), nn.ReLU()]
                for _ in range(max(global_layers - 2, 0)):
                    rho_layers.extend([nn.Linear(global_hidden, global_hidden), nn.ReLU()])
                rho_layers.append(nn.Linear(global_hidden, int(n_roi)))
                self.rho_net = nn.Sequential(*rho_layers)
            else:
                self.rho_net = None
        else:
            layers = max(int(layers), 2)
            net_layers = [nn.Linear(2, hidden), nn.ReLU()]
            for _ in range(layers - 2):
                net_layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
            net_layers.append(nn.Linear(hidden, 1))
            self.gamma_net = nn.Sequential(*net_layers)
            if use_rho:
                rho_layers = [nn.Linear(2, hidden), nn.ReLU()]
                for _ in range(layers - 2):
                    rho_layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
                rho_layers.append(nn.Linear(hidden, 1))
                self.rho_net = nn.Sequential(*rho_layers)
            else:
                self.rho_net = None

    def forward(self, T: torch.Tensor, t: torch.Tensor) -> LocalOutputs:
        if self.mode == "global":
            if T.dim() == 1:
                feats = torch.cat([T, t.view(1)], dim=0).unsqueeze(0)
                gamma = F.softplus(self.gamma_net(feats)).squeeze(0)
                rho = None
                if self.use_rho and self.rho_net is not None:
                    rho = self.rho_max * torch.tanh(self.rho_net(feats)).squeeze(0)
            else:
                if t.dim() == 0:
                    t_vec = t.expand(T.shape[0], 1)
                else:
                    t_vec = t.view(-1, 1)
                feats = torch.cat([T, t_vec], dim=1)
                gamma = F.softplus(self.gamma_net(feats))
                rho = None
                if self.use_rho and self.rho_net is not None:
                    rho = self.rho_max * torch.tanh(self.rho_net(feats))
            f_local = -gamma * T
            if rho is not None:
                f_local = f_local + rho * T * (1.0 - T)
            return LocalOutputs(f_local=f_local, gamma=gamma, rho=rho)

        if T.dim() == 1:
            t_vec = t.expand_as(T)
            feats = torch.stack([T, t_vec], dim=1)
            gamma = F.softplus(self.gamma_net(feats).squeeze(1))
            rho = None
            if self.use_rho and self.rho_net is not None:
                rho = self.rho_max * torch.tanh(self.rho_net(feats).squeeze(1))
            f_local = -gamma * T
            if rho is not None:
                f_local = f_local + rho * T * (1.0 - T)
            return LocalOutputs(f_local=f_local, gamma=gamma, rho=rho)
        t_mat = t.unsqueeze(1).expand_as(T)
        feats = torch.stack([T, t_mat], dim=2).reshape(-1, 2)
        gamma = F.softplus(self.gamma_net(feats).squeeze(1)).reshape(T.shape)
        rho = None
        if self.use_rho and self.rho_net is not None:
            rho = self.rho_max * torch.tanh(self.rho_net(feats).squeeze(1)).reshape(T.shape)
        f_local = -gamma * T
        if rho is not None:
            f_local = f_local + rho * T * (1.0 - T)
        return LocalOutputs(f_local=f_local, gamma=gamma, rho=rho)
