# AWS remote trainer instance sizing

## Decision

Ask the user which application-supported policy they plan to train, then map
that choice to an EC2 instance type. This avoids charging every user for the
largest possible GPU.

| Application policy | Instance type | GPU | GPU memory | Reason |
| --- | --- | --- | --- | --- |
| ACT | `g4dn.xlarge` | NVIDIA T4 | 16 GB | Small ResNet and transformer policy |
| SmolVLA | `g4dn.xlarge` | NVIDIA T4 | 16 GB | 500M backbone with expert-only training by default |
| Pi0.5 | `g6e.2xlarge` | NVIDIA L40S | 48 GB | Full fine-tuning is enabled in the application default |

## Repository evidence

- The application trainer accepts only `act`, `pi05`, and `smolvla`.
  See `application/backend/src/trainer/schemas.py`.
- Pi0.5 does not freeze the vision encoder and does not use expert-only training
  by default. See `library/configs/physicalai/pi05/aloha/default.yaml`.
- SmolVLA uses expert-only training and freezes its vision encoder by default.
  See `library/src/physicalai/policies/smolvla/config.py`.

## AWS evidence

- AWS documents the GPU and memory specifications for G4dn, G5, and G6e:
  [EC2 accelerated computing specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html).
- The template resolves the current recommended AL2023 GPU AMI, which includes
  NVIDIA drivers and the Docker GPU runtime:
  [Amazon ECS GPU workloads](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html).

## Deployment caveats

- Capacity and service quota are region- and Availability Zone-specific. Confirm
  the selected instance is offered in the target Availability Zone and request
  the required EC2 quota before deploying.
- G5 and G6e hosts use x86-64 AMD EPYC CPUs. The training accelerator remains an
  NVIDIA GPU, and the CUDA container is x86-64 compatible.
- The conclusion is based on model architecture and repository guidance, not a
  measured peak-memory run for every policy. Reducing batch size remains the
  first fallback if a workload exceeds the selected GPU memory.
