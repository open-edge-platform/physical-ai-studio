# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run Groot model on Intel XPU."""

import torch

from getiaction.policies.groot import GrootModel, get_best_device

print("=" * 60)
print("Running Groot Model on Intel XPU")
print("=" * 60)

device = get_best_device()
print(f"\nUsing device: {device}")

if device.type == "xpu":
    print(f"XPU device: {torch.xpu.get_device_name(0)}")
    props = torch.xpu.get_device_properties(0)
    print(f"XPU memory: {props.total_memory / 1024**3:.2f} GB")

print("\nLoading Groot model...")
model = GrootModel.from_pretrained(attn_implementation="sdpa")
model = model.to(device)
model.eval()

print("\n✓ Model loaded successfully!")
print(f"  - Backbone: {type(model.backbone).__name__}")
print(f"  - Action head: {type(model.action_head).__name__}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  - Total parameters: {total_params / 1e9:.2f}B")
print(f"  - Trainable parameters: {trainable_params / 1e6:.2f}M")

print("\n" + "=" * 60)
print("Groot model ready on XPU!")
print("=" * 60)
