# _*_ coding : utf-8 _*_
# @Time : 2026/3/17 14:35
# @Author : phy013x
# @File : model_lora.py

import torch
from torch import nn, optim

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank)
        self.B = nn.Linear(rank, out_features)

        self.A.weight.data.normal_(0, 0.01)
        self.B.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank).to(module.device)

            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

def load_lora(model, path):
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {
                f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)








