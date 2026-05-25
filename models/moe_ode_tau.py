from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_clearance import LocalClearance, LocalOutputs


@dataclass
class MoEOutputs:
    traj: torch.Tensor
    f_phys: torch.Tensor
    f_local: torch.Tensor
    beta: torch.Tensor
    dT_total: torch.Tensor
    gamma: torch.Tensor
    rho: torch.Tensor | None
    d_reg: torch.Tensor
    d_smooth: torch.Tensor
    d_mean: torch.Tensor


class GateMLP(nn.Module):
    """Two-expert gate over physical and local dynamics."""

    def __init__(self, hidden: int = 16, use_mean: bool = True) -> None:
        super().__init__()
        self.use_mean = use_mean
        in_dim = 2 if use_mean else 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        if T.dim() == 1:
            mean_t = torch.mean(T).unsqueeze(0)
        else:
            mean_t = torch.mean(T, dim=1, keepdim=True)
        feats = torch.cat([t.unsqueeze(-1), mean_t], dim=-1) if self.use_mean else t.unsqueeze(-1)
        return F.softmax(self.net(feats), dim=-1)


class DiffusionScaler(nn.Module):
    def __init__(self, hidden: int = 32, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = float(eps)
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, T: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if T.dim() == 1:
            t_vec = t.expand_as(T)
            feats = torch.stack([T, t_vec], dim=1)
            return F.softplus(self.net(feats).squeeze(1)) + self.eps
        t_mat = t.unsqueeze(1).expand_as(T)
        feats = torch.stack([T, t_mat], dim=2).reshape(-1, 2)
        return F.softplus(self.net(feats).squeeze(1)).reshape(T.shape) + self.eps


class MoEODETau(nn.Module):
    def __init__(
        self,
        n_roi: int,
        L_sc: torch.Tensor,
        reaction: str = "logistic",
        gate_use_mean: bool = True,
        enable_local: bool = True,
        phys_dmod: bool = False,
        local_hidden: int = 32,
        local_use_rho: bool = False,
        local_rho_max: float = 0.1,
        local_mode: str = "per_roi",
        local_global_hidden: int = 64,
        local_global_layers: int = 2,
        local_layers: int = 2,
        phys_d_hidden: int = 32,
        phys_d_eps: float = 1e-3,
        disable_gate: bool = False,
        fixed_beta: list[float] | tuple[float, ...] | None = None,
        fixed_beta_normalize: bool = True,
        force_k_zero: bool = False,
        force_r_zero: bool = False,
    ) -> None:
        super().__init__()
        self.n_roi = n_roi
        self.L_sc = L_sc
        self.reaction = reaction
        self.enable_local = enable_local
        self.local_expert = LocalClearance(
            n_roi=n_roi,
            hidden=local_hidden,
            use_rho=local_use_rho,
            rho_max=local_rho_max,
            mode=local_mode,
            global_hidden=local_global_hidden,
            global_layers=local_global_layers,
            layers=local_layers,
        )
        self.gate = GateMLP(use_mean=gate_use_mean)
        self.disable_gate = bool(disable_gate)
        self.force_k_zero = bool(force_k_zero)
        self.force_r_zero = bool(force_r_zero)
        self.fixed_beta: torch.Tensor | None = None
        self.fixed_beta_normalize = bool(fixed_beta_normalize)
        if fixed_beta is not None:
            fixed = torch.tensor([float(x) for x in fixed_beta], dtype=torch.float32)
            if fixed.numel() != 2:
                raise ValueError("fixed_beta must be [beta_phys, beta_local].")
            if torch.any(fixed < 0):
                raise ValueError("fixed_beta values must be non-negative.")
            if float(torch.sum(fixed).item()) <= 0:
                raise ValueError("fixed_beta must contain at least one positive value.")
            self.fixed_beta = fixed
        self.k_raw = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.r_raw = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.phys_dmod = bool(phys_dmod)
        self.d_mod = DiffusionScaler(hidden=phys_d_hidden, eps=phys_d_eps) if self.phys_dmod else None

    def constrained_k(self, k_max: float | None = None) -> torch.Tensor:
        if self.force_k_zero:
            return torch.tensor(0.0, dtype=self.k_raw.dtype, device=self.k_raw.device)
        if k_max is None:
            return F.softplus(self.k_raw)
        return k_max * torch.sigmoid(self.k_raw)

    def reaction_term(self, T: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.reaction == "linear":
            return r * T
        return r * T * (1.0 - T)

    def f_phys(self, T: torch.Tensor, k: torch.Tensor, r: torch.Tensor, L_phys: torch.Tensor) -> torch.Tensor:
        diff = -k * (L_phys @ T) if T.dim() == 1 else -k * (T @ L_phys.T)
        return diff + self.reaction_term(T, r)

    def f_local(self, T: torch.Tensor, t: torch.Tensor) -> LocalOutputs:
        return self.local_expert(T, t)

    def rhs(
        self,
        T: torch.Tensor,
        t: torch.Tensor,
        k: torch.Tensor,
        r: torch.Tensor,
        L_phys: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if L_phys is None:
            L_phys = self.L_sc
        f_m = self.f_phys(T, k, r, L_phys)
        if self.enable_local:
            local_out = self.f_local(T, t)
            f_l = local_out.f_local
            gamma = local_out.gamma
            rho = local_out.rho
        else:
            f_l = torch.zeros_like(f_m)
            gamma = torch.zeros_like(f_m)
            rho = None

        if self.fixed_beta is not None:
            beta = self.fixed_beta.to(dtype=f_m.dtype, device=f_m.device)
            if T.dim() > 1:
                beta = beta.unsqueeze(0).expand(T.shape[0], -1)
        elif self.disable_gate:
            beta = torch.tensor([1.0, 1.0], dtype=f_m.dtype, device=f_m.device)
            if T.dim() > 1:
                beta = beta.unsqueeze(0).expand(T.shape[0], -1)
        else:
            beta = self.gate(t, T)

        if self.fixed_beta is not None or self.disable_gate or not self.enable_local:
            mask = torch.tensor([1.0, 1.0 if self.enable_local else 0.0], dtype=beta.dtype, device=beta.device)
            if beta.dim() == 1:
                beta = beta * mask
                if self.fixed_beta_normalize or self.disable_gate:
                    beta = beta / (torch.sum(beta) + 1e-8)
            else:
                beta = beta * mask
                if self.fixed_beta_normalize or self.disable_gate:
                    beta = beta / (torch.sum(beta, dim=1, keepdim=True) + 1e-8)

        if beta.dim() == 1:
            rhs = beta[0] * f_m + beta[1] * f_l
        else:
            rhs = beta[:, 0:1] * f_m + beta[:, 1:2] * f_l
        return rhs, f_m, f_l, beta, gamma, rho, rhs

    def rk4_step(
        self,
        T: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        k: torch.Tensor,
        r: torch.Tensor,
        L_phys: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        k1, f_m1, f_l1, b1, g1, rho1, rhs1 = self.rhs(T, t, k, r, L_phys)
        k2, f_m2, f_l2, b2, g2, rho2, rhs2 = self.rhs(T + 0.5 * dt * k1, t + 0.5 * dt, k, r, L_phys)
        k3, f_m3, f_l3, b3, g3, rho3, rhs3 = self.rhs(T + 0.5 * dt * k2, t + 0.5 * dt, k, r, L_phys)
        k4, f_m4, f_l4, b4, g4, rho4, rhs4 = self.rhs(T + dt * k3, t + dt, k, r, L_phys)
        T_next = T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        f_m = (f_m1 + 2 * f_m2 + 2 * f_m3 + f_m4) / 6.0
        f_l = (f_l1 + 2 * f_l2 + 2 * f_l3 + f_l4) / 6.0
        beta = (b1 + 2 * b2 + 2 * b3 + b4) / 6.0
        rhs = (rhs1 + 2 * rhs2 + 2 * rhs3 + rhs4) / 6.0
        gamma = (g1 + 2 * g2 + 2 * g3 + g4) / 6.0
        rho = None
        if rho1 is not None:
            rho = (rho1 + 2 * rho2 + 2 * rho3 + rho4) / 6.0
        return T_next, f_m, f_l, beta, rhs, gamma, rho

    def integrate(
        self,
        T0: torch.Tensor,
        t_grid: torch.Tensor,
        dt: float,
        k_max: float | None = None,
        safety_clip: bool = False,
        safety_min: float = -0.1,
        safety_max: float = 1.1,
    ) -> MoEOutputs:
        k = self.constrained_k(k_max)
        r = torch.tensor(0.0, dtype=self.r_raw.dtype, device=self.r_raw.device) if self.force_r_zero else F.softplus(self.r_raw)

        traj = [T0]
        f_m_list = []
        f_l_list = []
        beta_list = []
        rhs_list = []
        gamma_list = []
        rho_list = []
        d_reg_accum = torch.tensor(0.0, dtype=T0.dtype, device=T0.device)
        d_smooth_accum = torch.tensor(0.0, dtype=T0.dtype, device=T0.device)
        d_mean_accum = torch.tensor(0.0, dtype=T0.dtype, device=T0.device)
        d_count = 0
        d_smooth_count = 0
        prev_d = None
        T = T0
        for idx in range(len(t_grid) - 1):
            if self.phys_dmod and self.d_mod is not None:
                d = self.d_mod(T, t_grid[idx])
                L_phys = (d[:, None] * self.L_sc) * d[None, :]
                d_reg_accum = d_reg_accum + torch.mean((d - 1.0) ** 2)
                d_mean_accum = d_mean_accum + torch.mean(d)
                d_count += 1
                if prev_d is not None:
                    d_smooth_accum = d_smooth_accum + torch.mean((d - prev_d) ** 2)
                    d_smooth_count += 1
                prev_d = d
            else:
                L_phys = self.L_sc
            T, f_m, f_l, beta, rhs, gamma, rho = self.rk4_step(T, t_grid[idx], dt, k, r, L_phys)
            if safety_clip:
                T = torch.clamp(T, safety_min, safety_max)
                T = torch.nan_to_num(T, nan=0.0, posinf=safety_max, neginf=safety_min)
            traj.append(T)
            f_m_list.append(f_m)
            f_l_list.append(f_l)
            beta_list.append(beta)
            rhs_list.append(rhs)
            gamma_list.append(gamma)
            if rho is not None:
                rho_list.append(rho)

        return MoEOutputs(
            traj=torch.stack(traj, dim=0),
            f_phys=torch.stack(f_m_list, dim=0) if f_m_list else torch.zeros((0, self.n_roi)),
            f_local=torch.stack(f_l_list, dim=0) if f_l_list else torch.zeros((0, self.n_roi)),
            beta=torch.stack(beta_list, dim=0) if beta_list else torch.zeros((0, 2)),
            dT_total=torch.stack(rhs_list, dim=0) if rhs_list else torch.zeros((0, self.n_roi)),
            gamma=torch.stack(gamma_list, dim=0) if gamma_list else torch.zeros((0, self.n_roi)),
            rho=torch.stack(rho_list, dim=0) if rho_list else None,
            d_reg=d_reg_accum / max(d_count, 1),
            d_smooth=d_smooth_accum / max(d_smooth_count, 1),
            d_mean=d_mean_accum / max(d_count, 1),
        )

    def extra_state(self, k_max: float | None = None) -> Dict[str, float]:
        r_val = 0.0 if self.force_r_zero else float(F.softplus(self.r_raw).detach().cpu().item())
        return {
            "k": float(self.constrained_k(k_max).detach().cpu().item()),
            "r": r_val,
        }
