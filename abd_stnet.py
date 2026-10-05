"""Adaptive Baseline-Deviation Spatio-Temporal Network (ABD-STNet)."""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ABDSTNet"]


def probability_to_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class HistoricalStatistics(nn.Module):
    def __init__(self, input_length: int):
        super().__init__()
        self.input_length = int(input_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,H,W], raw counts
        b, t, h, w = x.shape
        n = h * w
        raw = x.reshape(b, t, n)
        z = torch.log1p(torch.clamp(raw, min=0.0))
        last = z[:, -1]
        mean = z.mean(dim=1)
        maxv = z.max(dim=1).values
        std = z.std(dim=1, unbiased=False)
        nonzero_ratio = (raw > 0).float().mean(dim=1)
        mid = max(1, t // 2)
        trend = z[:, mid:].mean(dim=1) - z[:, :mid].mean(dim=1)
        recent_len = min(3, t)
        recent_nonzero = (raw[:, -recent_len:] > 0).float().mean(dim=1)
        max_mean_gap = maxv - mean
        return torch.stack(
            [last, mean, maxv, std, nonzero_ratio, trend, recent_nonzero, max_mean_gap],
            dim=-1,
        )


def build_local_adjacency(grid_height: int, grid_width: int) -> torch.Tensor:
    """Row-normalized four-neighbor grid graph without self loops."""
    n = grid_height * grid_width
    adj = torch.zeros(n, n, dtype=torch.float32)
    for r in range(grid_height):
        for c in range(grid_width):
            i = r * grid_width + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid_height and 0 <= cc < grid_width:
                    adj[i, rr * grid_width + cc] = 1.0
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)


def topk_row_softmax(S_d: torch.Tensor, K_dyn: int) -> torch.Tensor:
    if K_dyn <= 0 or K_dyn >= S_d.size(-1):
        return F.softmax(S_d, dim=-1)
    topv, topi = torch.topk(S_d, k=K_dyn, dim=-1)
    masked = torch.full_like(S_d, float('-inf'))
    masked.scatter_(-1, topi, topv)
    return F.softmax(masked, dim=-1)


class NodeAdaptiveDecomposition(nn.Module):
    """Decompose log-counts with unconstrained node-specific temporal filters."""
    def __init__(self, num_nodes: int, K_s: int = 7):
        super().__init__()
        if K_s <= 0 or K_s % 2 == 0:
            raise ValueError(f'K_s must be a positive odd integer, got {K_s}')
        self.num_nodes = int(num_nodes)
        self.K_s = int(K_s)
        self.temporal_filter = nn.Conv1d(
            self.num_nodes,
            self.num_nodes,
            kernel_size=self.K_s,
            padding=self.K_s // 2,
            groups=self.num_nodes,
            bias=False,
        )
        with torch.no_grad():
            self.temporal_filter.weight.fill_(1.0 / self.K_s)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, h, w = x.shape
        n = h * w
        Z_t = torch.log1p(torch.clamp(x.reshape(b, t, n), min=0.0))
        Z_base = self.temporal_filter(Z_t.transpose(1, 2)).transpose(1, 2)
        Z_base = Z_base[:, :t]
        Z_dev = Z_t - Z_base
        return Z_t, Z_base, Z_dev


class WeeklyContext(nn.Module):
    def __init__(self, input_length: int, d_b: int, w: int = 7, dropout: float = 0.1):
        super().__init__()
        self.w = int(w)
        self.required_len = 2 * self.w
        if input_length < self.required_len:
            raise ValueError(
                f'WeeklyContext requires at least {self.required_len} steps, got {input_length}'
            )
        self.proj = nn.Sequential(
            nn.Linear(4, d_b),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_b, d_b),
            nn.LayerNorm(d_b),
        )

    def forward(self, Z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = Z_t[:, -self.required_len:, :]
        mu_old = z[:, :self.w, :].mean(dim=1)
        mu_recent = z[:, self.w:, :].mean(dim=1)
        Delta_w = mu_recent - mu_old
        feature = torch.stack([mu_old, mu_recent, Delta_w, Delta_w.abs()], dim=-1)
        C_w = self.proj(feature)
        weekly_means = torch.stack([mu_old, mu_recent], dim=-1)
        return C_w, weekly_means


class BaselineTemporalEncoder(nn.Module):
    def __init__(self, d_b: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(1, d_b, kernel_size=3, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(d_b, d_b, kernel_size=3, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_b)

    def forward(self, Z_base: torch.Tensor) -> torch.Tensor:
        # Z_base: [B,T,N]
        b, t, n = Z_base.shape
        z = Z_base.permute(0, 2, 1).reshape(b * n, 1, t)
        z = self.net(z)[:, :, -1]
        return self.norm(z).reshape(b, n, -1)


class BaselineBranch(nn.Module):
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        d_b: int,
        dropout: float = 0.1,
        g_b_init: float = 0.1,
        g_w_init: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = grid_height * grid_width
        self.temporal = BaselineTemporalEncoder(d_b=d_b, dropout=dropout)
        self.E_node = nn.Embedding(self.num_nodes, d_b)
        self.E_dow = nn.Embedding(7, d_b)
        self.register_buffer('node_ids', torch.arange(self.num_nodes), persistent=False)
        self.register_buffer(
            'A_loc', build_local_adjacency(grid_height, grid_width), persistent=False
        )
        self.context_norm = nn.LayerNorm(d_b)
        self.W_b_loc = nn.Linear(d_b, d_b)
        self.local_norm = nn.LayerNorm(d_b)
        self.ffn = nn.Sequential(
            nn.Linear(d_b, d_b * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_b * 2, d_b),
        )
        self.out_norm = nn.LayerNorm(d_b)
        self.beta_b = nn.Parameter(
            torch.tensor(probability_to_logit(g_b_init), dtype=torch.float32)
        )
        self.beta_w = nn.Parameter(
            torch.tensor(probability_to_logit(g_w_init), dtype=torch.float32)
        )
        self.head = nn.Sequential(
            nn.Linear(d_b, d_b),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_b, 1),
        )
        self.softplus = nn.Softplus()

    def forward(
        self,
        Z_base: torch.Tensor,
        target_dow: Optional[torch.Tensor],
        C_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        H_b_temp = self.temporal(Z_base)
        b = H_b_temp.size(0)
        if target_dow is None:
            target_dow = torch.zeros(b, dtype=torch.long, device=H_b_temp.device)
        target_dow = target_dow.long().to(H_b_temp.device) % 7

        E_node = self.E_node(self.node_ids).unsqueeze(0)
        E_dow = self.E_dow(target_dow).unsqueeze(1)
        g_w = torch.sigmoid(self.beta_w)
        H_b_ctx = self.context_norm(H_b_temp + E_node + E_dow + g_w * C_w)

        M_b_loc = torch.einsum('ij,bjd->bid', self.A_loc, H_b_ctx)
        g_b = torch.sigmoid(self.beta_b)
        H_b_loc = self.local_norm(H_b_ctx + g_b * self.W_b_loc(M_b_loc))
        H_b_out = self.out_norm(H_b_loc + self.ffn(H_b_loc))
        r = self.softplus(self.head(H_b_out).squeeze(-1))
        aux = {
            'H_b_ctx': H_b_ctx,
            'H_b_out': H_b_out,
            'g_b': g_b,
            'g_w': g_w,
        }
        return r, H_b_out, aux


class DeviationTemporalEncoder(nn.Module):
    def __init__(self, d_d: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(3, d_d, kernel_size=3, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(d_d, d_d, kernel_size=3, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Sequential(
            nn.Linear(d_d * 2, d_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_d, d_d),
            nn.LayerNorm(d_d),
        )

    def forward(self, Z_dev: torch.Tensor) -> torch.Tensor:
        # Z_dev: [B,T,N]
        b, t, n = Z_dev.shape
        dx = Z_dev[:, 1:, :] - Z_dev[:, :-1, :]
        dx = F.pad(dx, (0, 0, 1, 0))
        positive_dx = F.relu(dx)
        feat = torch.stack([Z_dev, dx, positive_dx], dim=-1)  # [B,T,N,3]
        feat = feat.permute(0, 2, 3, 1).reshape(b * n, 3, t)
        z = self.net(feat)
        h_last = z[:, :, -1]
        h_mean = z.mean(dim=-1)
        return self.out(torch.cat([h_last, h_mean], dim=-1)).reshape(b, n, -1)


class LocalDynamicPropagation(nn.Module):
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        d_d: int,
        K_dyn: int = 7,
        dropout: float = 0.1,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.K_dyn = int(K_dyn)
        self.d_d = int(d_d)
        self.d_a = self.d_d
        self.register_buffer(
            'A_loc', build_local_adjacency(grid_height, grid_width), persistent=False
        )
        self.W_Q = nn.Linear(d_d, d_d)
        self.W_K = nn.Linear(d_d, d_d)
        self.W_loc = nn.Linear(d_d, d_d)
        self.W_dyn = nn.Linear(d_d, d_d)
        self.W_O_loc = nn.Linear(d_d, d_d)
        self.W_O_dyn = nn.Linear(d_d, d_d)
        self.beta_loc = nn.Parameter(
            torch.tensor(probability_to_logit(init_scale), dtype=torch.float32)
        )
        self.beta_dyn = nn.Parameter(
            torch.tensor(probability_to_logit(init_scale), dtype=torch.float32)
        )
        self.norm1 = nn.LayerNorm(d_d)
        self.ffn = nn.Sequential(
            nn.Linear(d_d, d_d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_d * 2, d_d),
        )
        self.norm2 = nn.LayerNorm(d_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M_d_loc = torch.einsum('ij,bjd->bid', self.A_loc, self.W_loc(H_d))
        S_d = torch.matmul(self.W_Q(H_d), self.W_K(H_d).transpose(1, 2)) / math.sqrt(self.d_a)
        A_dyn = topk_row_softmax(S_d, self.K_dyn)
        M_d_dyn = torch.matmul(A_dyn, self.W_dyn(H_d))

        g_loc = torch.sigmoid(self.beta_loc)
        g_dyn = torch.sigmoid(self.beta_dyn)
        M_d_fuse = (
            g_loc * self.W_O_loc(M_d_loc)
            + g_dyn * self.W_O_dyn(M_d_dyn)
        )
        out = self.norm1(H_d + self.dropout(M_d_fuse))
        out = self.norm2(out + self.ffn(out))
        return out, g_loc, g_dyn


class DeviationBranch(nn.Module):
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        d_d: int,
        K_dyn: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        stat_dim = 8
        self.temporal = DeviationTemporalEncoder(d_d=d_d, dropout=dropout)
        self.stat_projection = nn.Sequential(
            nn.Linear(stat_dim, d_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_d, d_d),
            nn.LayerNorm(d_d),
        )
        self.pre_norm = nn.LayerNorm(d_d)
        self.propagation = LocalDynamicPropagation(
            grid_height=grid_height,
            grid_width=grid_width,
            d_d=d_d,
            K_dyn=K_dyn,
            dropout=dropout,
        )
        self.q_head = nn.Sequential(
            nn.Linear(d_d * 2 + stat_dim, d_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_d, 1),
            nn.Sigmoid(),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(d_d * 2 + stat_dim, d_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_d, d_d // 2),
            nn.GELU(),
            nn.Linear(d_d // 2, 1),
        )

    def forward(
        self,
        Z_dev: torch.Tensor,
        S_stat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        H_d_temp = self.temporal(Z_dev)
        H_d_stat = self.stat_projection(S_stat)
        H_d = self.pre_norm(H_d_temp + H_d_stat)
        H_d_prop, g_loc, g_dyn = self.propagation(H_d)
        # q is auxiliary and uses temporal/statistical features before propagation.
        q = self.q_head(torch.cat([H_d_temp, H_d_stat, S_stat], dim=-1)).squeeze(-1)
        delta = self.delta_head(torch.cat([H_d_prop, H_d_temp, S_stat], dim=-1)).squeeze(-1)
        aux = {
            'g_loc': g_loc,
            'g_dyn': g_dyn,
        }
        return delta, q, H_d_prop, aux


class ABDSTNet(nn.Module):
    """One-step count forecasting for a single event category.

    X_t: raw nonnegative counts, [B, T, H, W].
    target_dow: target-day weekday indices, [B], in {0, ..., 6}.
    Returns y_hat [B, H, W] and a dictionary of intermediate tensors.
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        input_length: int = 14,
        d_b: int = 32,
        d_d: int = 64,
        dropout: float = 0.1,
        K_s: int = 7,
        K_dyn: int = 7,
        w: int = 7,
        g_b_init: float = 0.1,
        g_w_init: float = 0.1,
    ):
        super().__init__()
        if min(grid_height, grid_width, d_b, d_d, w) < 1 or d_d < 2:
            raise ValueError("Grid dimensions, d_b and w must be positive; d_d >= 2.")
        if input_length < 2 * w:
            raise ValueError("input_length must cover at least two weekly windows.")
        if not 1 <= K_dyn <= grid_height * grid_width:
            raise ValueError("K_dyn must be between 1 and the number of nodes.")
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.num_nodes = self.grid_height * self.grid_width
        self.input_length = int(input_length)
        self.d_b = int(d_b)
        self.d_d = int(d_d)
        self.d_a = self.d_d

        self.decomposition = NodeAdaptiveDecomposition(self.num_nodes, K_s=K_s)
        self.statistics = HistoricalStatistics(input_length=input_length)
        self.weekly_context = WeeklyContext(
            input_length=input_length, d_b=self.d_b, w=w, dropout=dropout,
        )
        self.baseline_branch = BaselineBranch(
            grid_height=grid_height, grid_width=grid_width, d_b=self.d_b,
            dropout=dropout, g_b_init=g_b_init, g_w_init=g_w_init,
        )
        self.deviation_branch = DeviationBranch(
            grid_height=grid_height, grid_width=grid_width, d_d=self.d_d,
            K_dyn=K_dyn, dropout=dropout,
        )
        self.softplus = nn.Softplus()

    def forward(
        self, X_t: torch.Tensor, target_dow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if X_t.ndim != 4 or tuple(X_t.shape[1:]) != (
            self.input_length, self.grid_height, self.grid_width,
        ):
            raise ValueError("X_t must have shape [B, input_length, grid_height, grid_width].")
        if not X_t.is_floating_point():
            raise TypeError("X_t must be a floating-point tensor containing raw counts.")
        if target_dow is not None and target_dow.shape != (X_t.size(0),):
            raise ValueError("target_dow must have shape [B].")
        Z_t, Z_base, Z_dev = self.decomposition(X_t)
        S_stat = self.statistics(X_t)
        C_w, weekly_means = self.weekly_context(Z_t)
        r, H_b_out, base_aux = self.baseline_branch(
            Z_base, target_dow=target_dow, C_w=C_w,
        )
        delta, q, H_d_prop, dev_aux = self.deviation_branch(Z_dev, S_stat)
        y_hat = self.softplus(r + delta).reshape(
            X_t.size(0), self.grid_height, self.grid_width,
        )
        aux = {
            "r": r.reshape_as(y_hat),
            "delta": delta.reshape_as(y_hat),
            "q": q.reshape_as(y_hat),
            "Z_t": Z_t,
            "Z_base": Z_base,
            "Z_dev": Z_dev,
            "S_stat": S_stat,
            "H_b_out": H_b_out,
            "H_d_prop": H_d_prop,
            "C_w": C_w,
            "weekly_means": weekly_means,
            **base_aux,
            **dev_aux,
        }
        return y_hat, aux
