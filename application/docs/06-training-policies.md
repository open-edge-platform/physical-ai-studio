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

When training finishes we export the model to all its supported backends: [PyTorch](https://github.com/pytorch/pytorch), [OpenVINO](https://github.com/openvinotoolkit/openvino), [ONNX](https://github.com/onnx/onnx) and [ExecuTorch](https://github.com/pytorch/executorch).
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

- Run/deploy in UI: [Deploying Model Policies](./07-deploying-model-policies.md).
