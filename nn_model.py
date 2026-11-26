import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_t
from dataclasses import dataclass, field
from typing import Dict, Optional
from config import CFG
from odmr_core import reconstruct_spectrum_torch, N_LABELS


@dataclass
class ODMRPrediction:
    center_ghz: torch.Tensor
    split_mhz: torch.Tensor
    w1_mhz: torch.Tensor
    w2_mhz: torch.Tensor
    c1: torch.Tensor
    c2: torch.Tensor
    eta: torch.Tensor
    bl_offset: torch.Tensor
    bl_slope: torch.Tensor
    bl_quad: torch.Tensor
    log_sigma_sq: torch.Tensor
    raw: torch.Tensor

    def to_tensor(self) -> torch.Tensor:
        return torch.stack([
            self.center_ghz, self.split_mhz,
            self.w1_mhz, self.w2_mhz,
            self.c1, self.c2, self.eta,
            self.bl_offset, self.bl_slope, self.bl_quad,
        ], dim=-1)

    def get_sigma(self) -> torch.Tensor:
        return torch.exp(0.5 * self.log_sigma_sq)

    def get_variance(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_sq)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        d = {n: getattr(self, n) for n in
             ["center_ghz", "split_mhz", "w1_mhz", "w2_mhz",
              "c1", "c2", "eta", "bl_offset", "bl_slope", "bl_quad"]}
        d["log_sigma_sq"] = self.log_sigma_sq
        d["sigma"] = self.get_sigma()
        return d


class ODMR_CNN(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        use_bn = getattr(mc, "use_batchnorm", False)


        self.register_buffer(
            "log_var_max_per_param",
            torch.tensor(cfg.loss.log_var_max_per_param, dtype=torch.float32),
        )


        layers = []
        in_ch, length = 1, cfg.n_freq
        for cfg_tuple in mc.conv_layers:
            if len(cfg_tuple) == 3:
                out_ch, ks, stride = cfg_tuple
            elif len(cfg_tuple) == 2:
                out_ch, ks = cfg_tuple
                stride = 1
            else:
                raise ValueError(f"conv_layers entry must be (ch, k) or "
                                 f"(ch, k, stride), got {cfg_tuple}")
            layers.append(nn.Conv1d(in_ch, out_ch, ks, stride=stride))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(True))
            length = (length - ks) // stride + 1
            in_ch = out_ch
        self.convs = nn.Sequential(*layers)
        self._flat = in_ch * length
        print(f"[model] conv output length: {length}, flat: {self._flat}, "
              f"batchnorm: {use_bn}")


        # Fully connected layers.

        fc = []
        prev = self._flat
        for h in mc.fc_hidden:
            fc += [nn.Linear(prev, h), nn.ReLU(True)]
            if mc.dropout > 0:
                fc.append(nn.Dropout(mc.dropout))
            prev = h
        fc.append(nn.Linear(prev, mc.n_raw_outputs))
        self.fc = nn.Sequential(*fc)


        log_var_init = np.array([
            np.log(0.005 ** 2),
            np.log(5.0 ** 2),
            np.log(3.0 ** 2),
            np.log(3.0 ** 2),
            np.log(0.05 ** 2),
            np.log(0.05 ** 2),
            np.log(0.3 ** 2),
            np.log(0.05 ** 2),
            np.log(0.02 ** 2),
            np.log(0.01 ** 2),
        ], dtype=np.float32)
        with torch.no_grad():
            b = self.fc[-1].bias
            mean_init = torch.tensor([
                2.87,
                np.log(np.exp(8.0) - 1),
                np.log(np.exp(4.0) - 1),
                np.log(np.exp(4.0) - 1),
                np.log(np.exp(5.0) - 1),
                np.log(np.exp(5.0) - 1),
                0.0,
                1.0,
                0.0,
                0.0,
            ], dtype=torch.float32)
            b.data = torch.cat([
                mean_init,
                torch.from_numpy(log_var_init),
            ])
            self.fc[-1].weight.data.mul_(0.1)

    def forward(self, x: torch.Tensor) -> ODMRPrediction:
        h = self.convs(x)
        h = h.view(h.size(0), -1)
        r = self.fc(h)                                  # (B, 20)

        mean_raw = r[:, :10]

        log_var = r[:, 10:20].clamp(min=self.cfg.loss.log_var_min)
        log_var = torch.minimum(log_var, self.log_var_max_per_param.unsqueeze(0))

        return ODMRPrediction(
            center_ghz=mean_raw[:, 0],
            split_mhz=F_t.softplus(mean_raw[:, 1]),
            w1_mhz=F_t.softplus(mean_raw[:, 2]) + 0.5,
            w2_mhz=F_t.softplus(mean_raw[:, 3]) + 0.5,
            c1=F_t.softplus(mean_raw[:, 4]) * 0.01 + 1e-4,
            c2=F_t.softplus(mean_raw[:, 5]) * 0.01 + 1e-4,
            eta=torch.sigmoid(mean_raw[:, 6]),
            bl_offset=mean_raw[:, 7],
            bl_slope=mean_raw[:, 8],
            bl_quad=mean_raw[:, 9],
            log_sigma_sq=log_var,
            raw=r,
        )


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_model(cfg=CFG, device=None):
    if device is None:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = ODMR_CNN(cfg).to(device)
    print(f"[model] ODMR_CNN -- {count_parameters(model):,} params, "
          f"flat={model._flat}, output_dim={cfg.model.n_raw_outputs}")
    return model, device


_PARAM_NAMES = ["center", "split", "w1", "w2", "c1", "c2",
                "eta", "bl_off", "bl_sl", "bl_q"]


class ODMRLoss(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        lc = cfg.loss


        self.register_buffer(
            "nll_weights",
            torch.tensor(lc.nll_weights, dtype=torch.float32)
        )

        self.w_sp = lc.w_spectral
        self.do_spec = lc.enable_spectral_loss


        self.register_buffer(
            "freq_mhz",
            torch.from_numpy(cfg.freq_axis_ghz * 1e3).float()
        )

    def forward(self, pred: ODMRPrediction, labels: torch.Tensor,
                input_spec: Optional[torch.Tensor] = None):

        mu = pred.to_tensor().float()                    # (B, 10)
        log_var = pred.log_sigma_sq.float()              # (B, 10)
        labels_f = labels.float()                        # (B, 10)

        # Per-parameter NLL
        se = (mu - labels_f) ** 2                         # (B, 10)
        nll = 0.5 * (se * torch.exp(-log_var) + log_var)   # (B, 10)


        w = self.nll_weights.to(nll.device).unsqueeze(0)   # (1, 10)
        nll_weighted = nll * w                              # (B, 10)
        nll_total = nll_weighted.mean()

        ld = {
            "nll": nll_total.item(),
            "param_mse_unweighted": se.mean().item(),
            "log_var_mean": log_var.mean().item(),
        }


        with torch.no_grad():
            for i, n in enumerate(_PARAM_NAMES):
                ld[f"nll_{n}"] = nll[:, i].mean().item()
                ld[f"sigma_{n}"] = float(torch.exp(0.5 * log_var[:, i]).mean().item())

        total = nll_total


        if self.do_spec and input_spec is not None:
            recon = reconstruct_spectrum_torch(
                self.freq_mhz,
                pred.center_ghz, pred.split_mhz,
                pred.w1_mhz, pred.w2_mhz,
                pred.c1, pred.c2, pred.eta,
                pred.bl_offset, pred.bl_slope, pred.bl_quad,
            )
            rmu = recon.mean(dim=1, keepdim=True)
            rstd = recon.std(dim=1, keepdim=True).clamp(min=1e-6)
            recon_z = (recon - rmu) / rstd
            sp_loss = F_t.l1_loss(recon_z, input_spec.squeeze(1))
            total = total + self.w_sp * sp_loss
            ld["spectral"] = sp_loss.item()

        ld["total"] = total.item()
        return total, ld
