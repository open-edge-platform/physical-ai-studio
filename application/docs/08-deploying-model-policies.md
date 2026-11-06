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

## Run a policy on a simulation from your own script

A simulation that Studio starts stops when Studio stops. To drive a simulated robot from your own script, start its simulation yourself:

1. Open the robot's start simulation dialog and pick the scene, and a seed under Advanced if you want one.
2. Open **Run it yourself** and copy the command.
3. In a terminal on the machine Studio runs on, activate the Python environment Studio runs in: `source .venv/bin/activate` in `backend` for a source checkout, or the environment you installed Studio into with pip.
4. Run the command. The simulation runs until you stop it with Ctrl+C.

Studio connects to it as it is: the robot shows `Running · outside Studio`, with its 3D viewer and cameras, and the start dialog, if it is still open, says so.

To run a trained policy on it from a script:

1. On the Models screen, open the model and click **Runtime export** on its OpenVINO format.
2. Pick an environment with that simulated robot and a device, and download the zip.
3. Unzip it and, with the same Python environment active (`runtime.yaml` loads the simulation plugin from it), run the policy from that folder:

```bash
physicalai run --config runtime.yaml --run.duration_s=60
```

Limits:

- This works with a native install only. A simulation accepts robot connections from its own machine only, so Studio in Docker can't reach a simulation started on the host, and a script on the host can't reach one that Studio started in its container.
- Nothing stops Studio and your script from driving the robot at the same time. Before you run a script, stop Studio's session on the robot and leave its environment page: the session keeps the robot for about 45 seconds after you leave.
- Keep the port the robot is configured with. The command uses the robot's name and address, and Studio's cameras and `runtime.yaml` point at that address. If you change the port in the command, change the robot's address in Studio to match. A simulation that Studio starts reuses that port too. If another program has it, Studio starts the simulation on a free port, says so in the start dialog, and moves the robot and its cameras to that port, so the **Run it yourself** command and new runtime exports use it from then on.

## Next

- For training, see [Training Policies](./06-training-policies.md).
