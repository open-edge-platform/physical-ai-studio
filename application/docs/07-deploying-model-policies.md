# Deploying Model Policies

| **Run model policy**               | **Model inference screen**                  |
|------------------------------------|---------------------------------------------|
| ![Run model policy][inference-run] | ![Model inference screen][inference-screen] |

[inference-run]: ./assets/07-inference-run.png
[inference-screen]: ./assets/07-inference-screen.png

Models can run inside Physical AI Studio or be deployed with [OpenVINO PhysicalAI](https://github.com/openvinotoolkit/physicalai).
On the Models screen, click Run model for a trained policy, then select the inference backend and device. Studio currently supports PyTorch and OpenVINO on GPU or CPU.

When you start the model, Studio loads the environment used to record the dataset the model was trained on.
After the environment and model load, you will see a screen similar to the recording view. Pick the task the model should perform and click Play.

## Next

- For training, see [Training Policies](./06-training-policies.md).
