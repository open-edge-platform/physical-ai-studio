# Training Policies

| **Train model policy**                           | **Open model training logs**             | **Model formats**                                  |
|--------------------------------------------------|------------------------------------------|----------------------------------------------------|
| ![Train model policy][models-train-model-policy] | ![Open model training logs][model-logs]  | ![Download optimized model formats][model-formats] |

[models-train-model-policy]: ./assets/06-models-train.png
[model-logs]: ./assets/06-model-logs.png
[model-formats]: ./assets/06-models-formats.png

This guide describes how users train models from the Models page.

## Train a new model policy

Once you've collected enough episodes for your dataset you can begin to train a new model policy.
First, choose the model policy. We currently support:

- ACT
- SmolVLA
- Pi0.5

Some policies download assets from Hugging Face Hub during setup or training. Configure `HF_TOKEN` before training Hub-backed policies such as SmolVLA or Pi0.5, especially on shared networks or when using gated/private models.

Depending on the amount of VRAM available on your GPU, you may need to adjust the advanced settings.
These settings include _batch size_, _training steps_, _amount of data workers_, _precision_, and an option to _compile model_ before training.
You may need to tune these settings to get an optimal result.

## Hugging Face Hub access

If `HF_TOKEN` is not set, the backend uses unauthenticated Hugging Face Hub access and may log a warning. Downloads can fail without a token because of anonymous rate limits or access restrictions on gated/private repositories.

Use a token with read-only model access:

- Required: `Read` permission for model repositories.
- Not required: `Write` or admin permissions.
- For gated/private models, use a Hugging Face account that has access to those repositories.
- For fine-grained tokens, grant read access to the specific model repositories you plan to use.

To create a token:

1. Sign in to [huggingface.co](https://huggingface.co/).
2. Open **Settings** -> **Access Tokens**.
3. Create a new token.
4. Set permissions to read-only model access.
5. Copy the token value.

For Docker deployments, add the token to `application/docker/.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Then recreate or start the Docker stack from `application/docker/`:

```bash
docker compose up -d --force-recreate
```

For native backend deployments, add the token to `application/backend/.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Then start the backend from `application/backend/`:

```bash
./run.sh
```

Never commit real tokens to source control. Store them only in local `.env` files or your secret manager, and rotate the token immediately if it is exposed.

## Remote training

Run training on a separate GPU server while you record datasets on a lightweight machine. Studio supports two training modes, set by the `TRAINING_MODE` environment variable on the backend:

- `local` (default): training runs in the same process as the backend. This requires the `[train]` dependency extra (torch, transformers, ExecuTorch, and the policy weights).
- `remote`: training runs on a separate **trainer service**. The recording install stays lightweight and does not need the `[train]` extra.

Use `remote` mode when the machine you record on lacks a capable GPU, or when you want one GPU server to serve several recording stations.

The Studio experience is unchanged in `remote` mode. You start training from the Models screen with the same form, and job status, progress, and the live loss curve appear exactly as for local training. Studio shows the dataset upload as an early progress step of the same job. Studio ignores the device you select in the form for remote jobs — the trainer server picks its own accelerator.

### Enable remote mode on the backend

Set these variables on the Studio backend. `TRAINING_MODE=remote` requires `TRAINER_URL`; the backend fails to start with a validation error if it is missing.

| Variable | Required | Description |
|----------|----------|-------------|
| `TRAINING_MODE` | yes | Set to `remote` to offload training. |
| `TRAINER_URL` | yes (remote) | Base URL of the trainer service, e.g. `http://trainer.internal:8001`. Use HTTPS only when a reverse proxy or TLS terminator serves the trainer. |
| `TRAINER_DATASET_TRANSFER` | no | Dataset transfer mode: `http` (default) streams a ZIP directly to the trainer; `hf` uses a temporary private Hugging Face dataset repository. |
| `TRAINER_HF_NAMESPACE` | no | Hugging Face org or user namespace for temporary dataset repositories; used only with `TRAINER_DATASET_TRANSFER=hf`. |
| `TRAINER_REQUEST_TIMEOUT_S` | no | HTTP timeout for non-streaming trainer calls (default `30`). |

For the default HTTP transfer, add these variables to `application/backend/.env`:

```env
TRAINING_MODE=remote
TRAINER_URL=http://trainer.internal:8001
```

Then start the backend from `application/backend/`:

```bash
./run.sh
```

For Docker deployments, add the same variables to `application/docker/.env`, then recreate the stack from `application/docker/`:

```bash
docker compose up -d --force-recreate
```

The trainer service must be running and reachable at `TRAINER_URL` before you start a remote job. See [Remote Training Server](./08-remote-training-server.md) to set it up.

For optional HF transfer, also set `TRAINER_DATASET_TRANSFER=hf` and `TRAINER_HF_NAMESPACE`; see [Hugging Face token requirements for remote mode](#hugging-face-token-requirements-for-remote-mode).

### Hugging Face token requirements for remote mode

Remote mode streams each dataset snapshot directly from the backend to the trainer over HTTP by default. The upload is resumable and the trainer removes its working copy after the job finishes. This default transfer does not require Hugging Face credentials for the dataset transfer itself.

Set `TRAINER_DATASET_TRANSFER=hf` only when you need to transfer snapshots through Hugging Face Hub. For every such job, the backend pushes the snapshot to a new temporary private dataset repository, pins its exact commit, and the trainer pulls the snapshot from that pinned commit. Studio deletes the temporary repository after it imports the trained model, including on failure.

This reuses the same `HF_TOKEN` described above, but the access level differs:

- **Local mode** needs **read** access when it downloads Hub-backed policy weights.
- **Remote mode with the default `http` transfer** does not need a token for dataset transfer. The backend and trainer still need the appropriate access for any Hub-backed policy assets they download.
- **Remote mode with `TRAINER_DATASET_TRANSFER=hf`** needs **write** access on the backend. The backend creates, uploads to, and deletes private dataset repositories under your namespace, so the token must allow creating and deleting repos, not just writing content.

With `TRAINER_DATASET_TRANSFER=hf`, the trainer service needs an `HF_TOKEN` with **read** access to pull those snapshots. Set tokens through the environment only and never commit them.

For exact classic and fine-grained token permissions, see [Hugging Face Integration](../backend/docs/huggingface_integration.md#required-token-permissions).

## Monitor training progress

| **Training job in progress**                              | **Open model training logs**             |
|-----------------------------------------------------------|------------------------------------------|
| ![Training job in progress][model-train-job-in-progress]  | ![Open model training logs][model-logs]  |

[model-train-job-in-progress]: ./assets/06-model-train-job-in-progress.png

After you start training, you can see its progress in the Models screen. Click the job to see a live view of its loss curve.
You may also view the training logs.

If a training job takes too long, you can interrupt it. This stores a checkpoint of the current model and exports the model to deployable formats.

## Model formats

| **Model formats**                                  |
|----------------------------------------------------|
| ![Download optimized model formats][model-formats] |

When training finishes, Studio exports the model to the supported backends whose dependencies are installed: [PyTorch](https://github.com/pytorch/pytorch), [OpenVINO](https://github.com/openvinotoolkit/openvino), [ONNX](https://github.com/onnx/onnx), and [ExecuTorch](https://github.com/pytorch/executorch).
Download the model and then use [OpenVINO PhysicalAI](https://github.com/openvinotoolkit/physicalai) to deploy it on your hardware.

## Troubleshooting training network errors

If training fails with a network error such as `urlopen error [Errno 99] Cannot assign requested address`, verify that the running container has the expected proxy configuration. Some training paths and model policies may contact external services such as Hugging Face Hub.

From `application/docker/`, check the host `.env`, the rendered Compose configuration, and the running container environment:

```bash
grep -i proxy .env
docker compose config | grep -i proxy
```

If proxy values are present in `.env` but missing from `docker compose ... config` or from the running container, upgrade to Docker Compose v2.24.0+ and recreate the container:

```bash
docker compose up -d --force-recreate
```

## Next

- Set up remote training: [Remote Training Server](./08-remote-training-server.md).
- Run/deploy in UI: [Deploying Model Policies](./07-deploying-model-policies.md).
