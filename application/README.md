<p align="center">
  <img src="../docs/assets/physical_ai_studio.png" alt="Physical AI Studio Application" width="100%">
</p>

# Physical AI Studio Application

Studio application for collecting demonstration data and managing VLA model training.

The application provides a graphical interface to:

- **Setup** your robot arms and cameras
- **Collect** demonstration data from robotic systems
- **Manage** datasets and training configurations
- **Train** policies using the PhysicalAI library
- **Deploy** trained models to production

<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="../docs/assets/application.gif" alt="Application demo" width="100%">
</p>
<!-- markdownlint-enable MD033 -->

## Components

| Component                 | Description                                                   | Documentation                         |
| ------------------------- | ------------------------------------------------------------- | ------------------------------------- |
| **[Backend](./backend/)** | FastAPI server for data management and training orchestration | [Backend README](./backend/README.md) |
| **[UI](./ui/)**           | React web application                                         | [UI README](./ui/README.md)           |
| **[Docker](./docker/)**   | All-in-one containerized deployment                           | [Docker README](./docker/readme.md)   |

## Quick Start

Full setup instructions in component READMEs. Quick reference:

### Backend

```bash
cd backend
uv sync
source .venv/bin/activate
./run.sh
```

Backend runs at http://localhost:8000

### Frontend

```bash
cd ui
nvm use
npm install
# Download the OpenAPI spec and generate types (assumed backend is running at http://localhost:7860)
npm run build:api:download && npm run build:api 
# Start the acutal UI
npm run start
```

UI runs at http://localhost:3000

### Docker (recommended)

Run the full application (backend + UI) in a single container:

```bash
cd docker
cp .env.example .env
docker compose up
```

Application runs at http://localhost:7860. See the [Docker README](./docker/readme.md) for
hardware configuration (Intel XPU, NVIDIA CUDA) and device setup.

## Getting started

This guide will setup a project for imitation learning using teleoperation.
In this example we'll be using a SO-101.

### **0. Create a project.**

Physical AI Studio groups the robot problems into Projects. This project will house the datasets and models for the specific problem (e.g. Assemble Part Y). 

### **1. Setup your robot arms**

**Objective:** Connect and calibrate robot arms & cameras

In order to record datasets and train models we need to setup the environment.
This environment consists of robot arms and cameras. At first a SO101 robot arm will need to be configured. Physical AI Studio will recognize an already setup robot and skip those steps.

1. **Add Follower Robot**
  - [Image: Add follower robot arm]
  - Name the robot and select the robot type SO101 Follower.
  - Select the robot. The serial IDs are listed in the dropdown. 
  If you are unsure which serial ID is which robot, press the identify button to open and close the gripper.
  - Press Add Robot
  - Check if there are any issues and resolve them with help of the UI.
  
2. **Setup motors**
The SO101 daisy chains the servos. In order to know which servo is which joint it will need to assign an ID to the servo.

  - [Image: Setup motors]
  - Only connect a specific servo to the controller board and press *Assign ID*
  - Repeat for every motor
  - Reconnect all motors.
  - Press *Verify motors* to continue.
  
  
3. **Calibration**
The SO101 needs to know the root position and the servos range.
  - [Image: Calibration]
  - Move the robot arm to the displayed *center of its range of motion* and press *Apply Homing Offsets*.
  - Move each joint through its entire range of motion.
  - Press *Finish Recording*
  
4. **Verification**

  - [Image: Verification]
  - Quickly note the robot details
  - Press *Save Robot*
  
  
Repeat this process for the SO101 Leader.

### **2. Setup cameras**

**Objective:** Setup cameras that will be used by the follower.

1. **Add new camera**
  <img src="../docs/assets/getting_started/add_camera.png" alt="Add new camera" width="100%">
  - Select USB Camera.
  - Set camera name of video input of the model.
  - Select the camera from the list and check preview to verify correct camera.
  - Select resolution and FPS
  - Press *Add camera*
  
Repeat for all the points of view for the robot.

### **3. Setup environment**

**Objective:** Define environment for robot

This environment will define your robot and the cameras. This will be used in the dataset to determine the input features of the models.

1. **Configure new environment**
  <img src="../docs/assets/getting_started/new_environment.png" alt="Configure new environment" width="100%">
  - Press *Configure new enviroment*
  - Select previously defined follower.
  - Select the leader robot.
  - Press *Add*
  
2. **Add cameras**
  - Select cameras from list
  - Press *Add* 
  
3. **Verify**
  <img src="../docs/assets/getting_started/verify_environment.png" alt="Verify new environment" width="100%">
  - Press *Configure new enviroment*
  - A preview will be shown with the robots and camera point of views.
  - Verify the robots by moving them in real time.
  - Press *Add Environment*

### **4. Collect demonstration data from robot**

**Objective:** Create a dataset to train a model for your task.

1. **Create new dataset**
  <img src="../docs/assets/getting_started/new_dataset.png" alt="New Dataset" width="100%">
  - Press *New dataset*
  - Select the environment. This will determine the dataset features - and therefore the model features.
  - Select a name
  - Press *Save*
  
2. **Start recording**
  <img src="../docs/assets/getting_started/start_recording.png" alt="Start Recording" width="100%">
  - Press *New dataset*
  - Select the environment. Multiple environments are allowed and datasets can be recorded using different environments as long as they have the same *features*.
  - Name the task
  
3. **Start Episode**
  <img src="../docs/assets/getting_started/start_episode.png" alt="Start Episode" width="100%">
  - Move the leader around to verify. Check lighting of scene.
  - Reset the environment for your task.
  - Press *Start Episode*
  - Execute the movement
  - Press *Accept* if you're happy with the episode. Otherwise *Discard*.
  - Reset the environment and repeat until done
  - Press *< Adding Episode* in the banner to go back to the dataset and persist the new episodes.
  

### **5. Train policies using the PhysicalAI library**

**Objective:** Train a policy using the recorded dataset.
1. **Train model**
  <img src="../docs/assets/getting_started/train_model.png" alt="Train Model" width="100%">
  - Go to *Models* in the header
  - Press *Train model*
  - Fill in model name
  - Select Dataset
  - Select Policy
  - Press *Train*
  A model will be trained. It can be interrupted and it will still save. 


2. **Run Inference**
  <img src="../docs/assets/getting_started/run_inference.png" alt="Run Inference" width="100%">
  - Once model has been trained you can run the model on your environment for verification.
  - Press *Run model* on your trained model.
  - Verify environment and backend.
  - Press *Start*
  A view of the robot arm and the cameras will appear.
  - Press *Play* to start inference on the arm.


## See Also

- **[Library](../library/)** - Python SDK for programmatic usage
- **[Main Repository](../README.md)** - Project overview
