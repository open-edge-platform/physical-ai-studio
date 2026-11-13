# Junction 2026: Intel Physical AI Challenge with Physical AI Studio

**Challenge:** _Stop Simulating, Start Moving: See It. Detect It. Act on It._

Build a closed-loop robot system on an SO-101 arm that inspects tabletop objects (such as Duplo bricks), detects the defective ones, and physically sorts them. Everything runs on an Intel Core Ultra AI PC.

```text
PERCEIVE ──► DETECT ──► REASON ──► EXECUTE ──► OPTIMIZE
cameras      Anomalib    VLA policy  SO-101      OpenVINO on
                                     arm         CPU / iGPU / NPU
```

The exact objects, defects, and sorting rules are announced at kickoff. How you build the pipeline is up to you. This guide covers the setup and the tools; the design choices are yours.

## Contents

1. [Step 1: Set up the station](#step-1-set-up-the-station)
2. [Step 2: Set up Modal for GPU training](#step-2-set-up-modal-for-gpu-training)
3. [Step 3: Configure Studio](#step-3-configure-studio)
4. [Step 4: Detect anomalies with Anomalib](#step-4-detect-anomalies-with-anomalib)
5. [Step 5: Record demonstrations and train a policy](#step-5-record-demonstrations-and-train-a-policy)
6. [Step 6: Close the loop](#step-6-close-the-loop)
7. [Step 7: Optimize on Intel hardware](#step-7-optimize-on-intel-hardware)
8. [Judging rubric](#judging-rubric)

---

## Step 1: Set up the station

- [ ] **Physical AI Studio runs** at http://localhost:3000.

  - **Recommended: run it natively** in two terminals:

    ```bash
    # Terminal 1: backend (http://localhost:7860)
    cd application/backend
    cp .env.example .env
    uv run --extra xpu physicalai-studio serve
    ```

    ```bash
    # Terminal 2: UI (http://localhost:3000)
    cd application/ui
    npm install
    npm run start
    ```

  - **Alternative: Docker.** From `application/docker/`, run `./setup-devices.sh --xpu`, then `docker compose up -d`.

- [ ] **Arms are connected:** the SO-101 leader and follower are powered and connected through the USB hub.
- [ ] **Cameras are mounted firmly.** You have an overhead camera, a wrist camera, and a second external camera. Decide what each one is for.
- [ ] **Hugging Face token:** a read-only token is set in **Settings → General → Hugging Face**. Hub-backed policies such as SmolVLA and Pi0.5 need it to download weights.

## Step 2: Set up Modal for GPU training

Training a VLA policy needs an NVIDIA GPU. Studio trains on **Modal**, and **each participant gets $100 of Modal training credit**. Set this up early, because the first image build takes several minutes.

1. Create an account at [modal.com](https://modal.com) and redeem your credit as the organizers instruct.
2. Give the **Studio backend** your Modal credentials. Logging in through your browser is not enough.

   - **Native backend:** run this as the user that runs the backend:

     ```bash
     uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal setup
     ```

   - **Docker:** pass `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to the backend container. Never commit the token.

3. In Studio, open **Settings → Training Targets → Modal Provider** and click **Deploy and connect**.

**Keep an eye on your credit:** you choose the GPU (A10G, A100, or H100) for each training job, and [prices](https://modal.com/pricing) differ a lot. Running jobs keep billing even if Studio is closed. Each job can run for at most 24 hours.

## Step 3: Configure Studio

Create a **project**, add your **robots** and **cameras**, and combine them into an **environment**. The [Getting Started guide](application/docs/03-getting-started.md) walks through these screens.

## Step 4: Detect anomalies with Anomalib

Physical AI Studio does not do anomaly detection. Use [Anomalib](https://github.com/open-edge-platform/anomalib) for this ([documentation](https://anomalib.readthedocs.io/en/latest/), [example notebooks](https://github.com/open-edge-platform/anomalib/tree/main/examples/notebooks)).

Install it in its own virtual environment so its PyTorch version doesn't clash with Studio's:

```bash
uv venv anomalib-env && source anomalib-env/bin/activate
uv pip install "anomalib[openvino,xpu]"
```

**Models to consider:**

| Model           | Description                                                     |
| --------------- | --------------------------------------------------------------- |
| **PaDiM**       | Simplest; trains in about 1 minute                              |
| **PatchCore**   | Very good results; trains in 3–5 minutes                        |
| **Dinomaly**    | One of the state-of-the-art models, but needs a lot of training |
| **EfficientAd** | Not state of the art, but fast at inference                     |
| **FoundAD**     | New model, reportedly better for few-shot                       |

> **Train anomaly models on the laptop's Intel GPU (XPU).** They are small enough to train locally, so they don't need Modal or your GPU credit. See the Anomalib README for the XPU accelerator settings.

**Hints:**

- Most Anomalib models train on good images only. Think about what "good" has to cover so that normal objects aren't rejected.
- Anomalib models export to OpenVINO for fast inference on Intel hardware.

## Step 5: Record demonstrations and train a policy

Use Studio to record teleoperation episodes with the leader arm, then train a policy on them. Studio supports **ACT**, **SmolVLA**, and **Pi0.5**. See [Recording Datasets](application/docs/05-recording-datasets.md) and [Training Policies](application/docs/06-training-policies.md).

**Hints:**

- Some policies accept a language task and some don't. That matters for how your detection result reaches the robot.
- Judges score robustness to shifted objects and clutter. Your data decides how robust your policy is.
- Review your episodes before training. Bad demonstrations make a bad policy.
- Studio exports trained models to PyTorch, OpenVINO, ONNX, and ExecuTorch. You can test a model on the arm from the Models page with **Run model**.

## Step 6: Close the loop

Connect detection to action so the robot sorts objects without someone choosing what it should do. How you connect them is up to you.

Useful pieces:

- [OpenVINO PhysicalAI](https://github.com/openvinotoolkit/physicalai) runs exported Studio policies outside the UI (`InferenceModel(...)`).
- Anomalib's OpenVINO inferencer runs your anomaly model.

Build in a way to stop the arm safely.

## Step 7: Optimize on Intel hardware

The Core Ultra has a CPU, an integrated GPU (iGPU), and an NPU. Decide which model runs where, and **measure** the result: latency per inference step and per sorting cycle, power, and temperature. Judges want evidence, not claims.

## Judging rubric

| Criterion                  | Points  | What judges look at                                                                |
| -------------------------- | ------- | ---------------------------------------------------------------------------------- |
| Task Success Rate          | 25      | Accuracy, cycle time, and robustness to shifted objects and clutter                |
| Intel Tech Integration     | 25      | Offloading to the NPU/iGPU, and deployment with OpenVINO and the Robotics AI Suite |
| Dataset & Pipeline Quality | 20      | Clean teleoperation data and effective fine-tuning in Physical AI Studio           |
| System Efficiency          | 20      | Latency per inference step, thermal headroom, and power consumption on Core Ultra  |
| Creativity                 | 10      | Original ideas beyond the baseline                                                 |
| **Total**                  | **100** |                                                                                    |
