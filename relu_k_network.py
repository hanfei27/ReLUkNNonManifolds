
import math
import torch
import torch.nn as nn


def _relu_power_second_moment(k_minus_one: int) -> float:
    """Return E[(ReLU(X)^(k-1))^2] for X~N(0,1)."""
    m = 2 * k_minus_one
    # half-normal even moment: 2^{m/2-1} * Gamma((m+1)/2) / sqrt(pi)
    return 2 ** (m / 2 - 1) * math.gamma((m + 1) / 2) / math.sqrt(math.pi)


class ReLUk(nn.Module):
    def __init__(self, k: int, normalize: bool = True, target_var: float = 0.5,
                 clamp_value: float | None = None):
        super().__init__()
        if k < 2:
            raise ValueError("ReLUk requires k >= 2")
        self.k = int(k)
        self.clamp_value = float(clamp_value) if clamp_value is not None else None
        if normalize:
            moment = _relu_power_second_moment(self.k - 1)
            # keep the same second moment as ReLU (0.5) unless user overrides target_var
            target = float(target_var) if target_var is not None else 1.0
            self.scale = math.sqrt(target / moment)
        else:
            self.scale = 1.0

    def forward(self, x):
        pos = torch.relu(x)
        if self.clamp_value is not None:
            pos = torch.clamp(pos, max=self.clamp_value)
        out = pos ** (self.k - 1)
        if self.scale != 1.0:
            out = out * (out.new_tensor(self.scale) if out.requires_grad else self.scale)
        return out


def make_mlp(in_dim, out_dim, width=None, depth=None, k=None, normalize=True,
             target_var=0.5, layer_norm=False, clamp_value=None,
             layer_widths=None, activation_orders=None):
    """Create an MLP where widths / activation orders can vary per layer."""

    if layer_widths is not None:
        widths = [int(w) for w in layer_widths]
        if not widths:
            raise ValueError("layer_widths must contain at least one width")
        inferred_depth = len(widths) + 1
        # if caller supplied depth/width/k they are ignored in favour of per-layer specs
    else:
        if width is None or depth is None or k is None:
            raise ValueError("width, depth and k must be provided when layer_widths is None")
        widths = [int(width)] * max(int(depth) - 1, 1)
        inferred_depth = int(depth)

    if activation_orders is None:
        if k is None:
            raise ValueError("activation_orders not provided and base k is None")
        activation_orders = [int(k)] * len(widths)
    else:
        activation_orders = [int(v) for v in activation_orders]
        if len(activation_orders) < len(widths):
            if k is None:
                raise ValueError("activation_orders shorter than widths and base k is None")
            activation_orders = activation_orders + [int(k)] * (len(widths) - len(activation_orders))
        elif len(activation_orders) > len(widths):
            activation_orders = activation_orders[:len(widths)]

    layers = []

    def add_block(in_features, out_features, act_k, apply_activation=True):
        layers.append(nn.Linear(in_features, out_features))
        if layer_norm:
            layers.append(nn.LayerNorm(out_features))
        if apply_activation:
            layers.append(ReLUk(act_k, normalize=normalize, target_var=target_var,
                                clamp_value=clamp_value))

    prev_dim = in_dim
    for width_i, act_k in zip(widths, activation_orders):
        add_block(prev_dim, width_i, act_k=act_k, apply_activation=True)
        prev_dim = int(width_i)

    layers.append(nn.Linear(prev_dim, out_dim))
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net


class SphereNet(nn.Module):
    def __init__(self, width=128, depth=4, k=3, weight_clip=1.0,
                 normalize_activation=True, activation_target_var=0.5,
                 use_layer_norm=False, activation_clamp=None,
                 layer_widths=None, activation_orders=None,
                 input_dim: int = 3,
                 normalize_input: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.normalize_input = bool(normalize_input)
        self.net = make_mlp(self.input_dim, 1, width, depth, k,
                            normalize=normalize_activation,
                            target_var=activation_target_var,
                            layer_norm=use_layer_norm,
                            clamp_value=activation_clamp,
                            layer_widths=layer_widths,
                            activation_orders=activation_orders)
        self.weight_clip = weight_clip
    def forward(self, x):
        if self.normalize_input:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
        return self.net(x)
    def clip_weights_(self):
        if self.weight_clip is None: return
        with torch.no_grad():
            c = float(self.weight_clip)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.clamp_(-c, c)
                    if m.bias is not None: m.bias.clamp_(-c, c)
