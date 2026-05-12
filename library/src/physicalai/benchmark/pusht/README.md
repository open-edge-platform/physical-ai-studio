# PushT Benchmark

The PushT benchmark is derived from the Diffusion Policy paper by Columbia Engineering: https://diffusion-policy.cs.columbia.edu/.

Each PushT gym is base seeded with a value of 100000 and a max number of steps of 300. They then test 50 gyms of different seeds and repeat 3 times — a total of 150 episodes.

The imitation data can be found [at the columbia diffusion paper page](https://diffusion-policy.cs.columbia.edu/data/training/). HuggingFace also hosts an example in `LeRobotDataset` format [in a hugging face dataset](https://huggingface.co/datasets/lerobot/pusht).

## Example

```python
from physicalai.benchmark.pusht import PushTBenchmark
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT
from physicalai.train import Trainer

# Train
datamodule = LeRobotDataModule(repo_id="lerobot/pusht")
policy = ACT()
trainer = Trainer(max_epochs=100)
trainer.fit(policy, datamodule)

# Evaluate (paper protocol: seed=100000, 50 episodes)
policy = ACT.load_from_checkpoint("experiments/lightning_logs/version_0/checkpoints/last.ckpt")
benchmark = PushTBenchmark(num_envs=3)  # let's us average over three
results = benchmark.evaluate(policy)
print(f"Success rate: {results.overall_success_rate:.1f}%")
```

## Citation

```bibtex
@inproceedings{chi2023diffusionpolicy,
  title     = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author    = {Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year      = {2023}
}

@article{chi2024diffusionpolicy,
  title   = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author  = {Chi, Cheng and Xu, Zhenjia and Feng, Siyuan and Cousineau, Eric and Du, Yilun and Burchfiel, Benjamin and Tedrake, Russ and Song, Shuran},
  journal = {The International Journal of Robotics Research},
  year    = {2024}
}
```
