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

Depending on the amount of VRAM available on your GPU, you may need to adjust the advanced settings.
These settings include _batch size_, _training steps_, _amount of data workers_, _precision_, and an option to _compile model_ before training.
You may need to tune these settings to get an optimal result.

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

## Next

- Run/deploy in UI: [Deploying Model Policies](./07-deploying-model-policies.md).
