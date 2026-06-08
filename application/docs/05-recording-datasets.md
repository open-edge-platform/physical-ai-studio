# Recording Datasets

| **Create a new dataset**              | **Dataset overview**                        | **Dataset recording**                  |
|---------------------------------------|---------------------------------------------|----------------------------------------|
| ![Create a new dataset][dataset-new]  | ![Dataset overview][dataset-review-episode] | ![Record episodes][dataset-recording]  |

[dataset-new]: ./assets/05-datasets-new.png
[dataset-review-episode]: ./assets/05-dataset-review-episode.png
[dataset-recording]: ./assets/05-dataset-recording.png

This guide covers what users do in the UI to collect demonstration episodes.

## 1. Create a dataset

Create a dataset, give it a name, and optionally set a default task. The selected environment is used when recording episodes.

### Import a dataset

| **Upload dataset**                        | **Import dataset**                 |
|-------------------------------------------|------------------------------------|
| ![Upload dataset][datasets-import-upload] | ![Import dataset][datasets-import] |

[datasets-import-upload]: ./assets/05-dataset-import-upload.png
[datasets-import]: ./assets/05-dataset-import-dataset.png

Alternatively, you can import a [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) dataset or a dataset exported from Physical AI Studio.
After upload, provide a name and select the default task and environment.

## 2. Open recording mode

| **No episodes yet**                      | **Dataset recording**                       |
|------------------------------------------|---------------------------------------------|
| ![No episodes yet][dataset-no-episodes]  | ![Recording an episode][dataset-recording]  |

[dataset-no-episodes]: ./assets/05-no-episodes.png

Before you start recording episodes, make sure that both the follower and leader arms are free to move.
Once you start recording the follower arm will follow the same movements as the leader.

From your dataset page, start recording by clicking Add episode.
Once your environment has finished loading you will see your camera feeds as well as a visualization of your follower robot.

In the top right you can also see the total episodes recorded in your dataset.
We recommend recording at least 50 episodes before you start training a model.

Typically recording episodes is done in a loop:

1. Reset your physical scene.
2. Enter or confirm the task.
3. Click Start episode.
4. Perform the demonstration.
5. Click Accept or Discard.

Keyboard shortcuts in recording view:

- Right arrow: Start episode or Accept.
- Left arrow: Discard.

> [!TIP]
> You can resize and close the camera and robot panels.

## 3. Review datasets

| **Review an episode**                        | **Remove episode from dataset**                        | **Export dataset**                | **Rename dataset**                |
|----------------------------------------------|--------------------------------------------------------|-----------------------------------|-----------------------------------|
| ![Review an episode][dataset-review-episode] | ![Remove episode from dataset][dataset-remove-episode] | ![Export dataset][dataset-export] | ![Rename dataset][dataset-rename] |

[dataset-remove-episode]: ./assets/05-dataset-remove-episode.png
[dataset-export]: ./assets/05-dataset-export.png
[dataset-rename]: ./assets/05-dataset-rename.png

After you've finished recording your episodes you may review each individual episode.
Episodes can be replayed in the UI and a graph of the joint states is shown to help you find bad episodes.

### Export dataset

At any time you may export a dataset in [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) format.
You can import this back into Physical AI Studio on another system, use [physicalai-train](https://github.com/open-edge-platform/physical-ai-studio/tree/main/library) or [lerobot](https://huggingface.co/docs/lerobot/index) to train models using your dataset outside of the Studio.

### Rename datasets

Use Edit dataset name when:

- A name is too generic.
- You want clearer versioning for experiments.

Keep names easy to scan in tabs and model-training pickers.

## Next

- Start training in [Training Policies](./06-training-policies.md).
