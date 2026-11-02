# Pi0.5 SnapFlow on LIBERO

A Pi0.5 policy distilled with SnapFlow for 20,000 steps reached **97.55% success** across the four standard LIBERO suites, versus **97.05%** for its 10-step teacher. On an NVIDIA H100 PCIe, the student's median inference latency was **58.9 ms per action chunk**, versus **175.4 ms** for the teacher: **2.98× faster per chunk**. These are Torch export measurements, not robot control-loop timings.

| Policy              | Denoising steps | Spatial | Object |  Goal | LIBERO-10 | Overall (2,000 episodes) | Median latency (ms/chunk) |
| ------------------- | --------------: | ------: | -----: | ----: | --------: | -----------------------: | ------------------------: |
| Pi0.5 teacher       |              10 |   98.0% |  99.0% | 96.4% |     94.8% |     1,941/2,000 (97.05%) |                     175.4 |
| SnapFlow, 10k steps |               1 |   97.4% |  99.8% | 96.4% |     93.6% |     1,936/2,000 (96.80%) |                      58.4 |
| SnapFlow, 20k steps |               1 |   98.8% |  99.8% | 96.8% |     94.8% | **1,951/2,000 (97.55%)** |                  **58.9** |

The 20k student completed 10 more episodes than the teacher out of 2,000. That 0.5 percentage-point difference comes from one seeded evaluation; it does not establish a statistically significant improvement or predict performance on other seeds, robots, or hardware.

## Method

The teacher was [`lerobot/pi05_libero_finetuned`](https://huggingface.co/lerobot/pi05_libero_finetuned), resolved at revision `8e174154ef5f6c60a8da12ae99c303d8963138c1`. The student started from those weights and trained on [`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero), revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e`. SnapFlow was enabled from step 0. The VLM was frozen; the action expert and target-time heads were trained with `snapflow_alpha=0.5`, `snapflow_lambda=0.1`, batch size 8, and bf16 mixed precision.

Studio's `LiberoBenchmark` evaluated Spatial, Object, Goal, and LIBERO-10 with seed 42, 50 episodes per task, and no video recording. The overall score is the arithmetic mean across the four equally sized suites. This is a Studio gym evaluation, separate from the vla-evaluation-harness results elsewhere in the repository.

Latency was measured separately with Runtime's `InferenceLatencyBenchmark` on exported Torch models. It timed preprocessing, model execution, and postprocessing for 100 batch-one action chunks after 5 warmup iterations, using seeded synthetic inputs matching the export schema. The measurement excludes simulator stepping, rendering, networking, model loading, and robot I/O. Rollout FPS includes simulator work and is not used as an inference-latency measure. The run used Torch `2.11.0+cu128` and Runtime `0.2.1.dev4+g0ad4548`.

The run targeted 30k steps with a 30k-step LR decay schedule. Training stopped at step 27,647, after the 20k evaluation. No 30k checkpoint or result was produced.

## Artifacts and evaluation requirements

The gated [Daankrol/pi05-snapflow-libero Hub repository](https://huggingface.co/Daankrol/pi05-snapflow-libero) holds the [10k checkpoint](https://huggingface.co/Daankrol/pi05-snapflow-libero/blob/main/checkpoints/snapflow-step010000.ckpt), [20k checkpoint](https://huggingface.co/Daankrol/pi05-snapflow-libero/blob/main/checkpoints/snapflow-step020000.ckpt), [result JSON files](https://huggingface.co/Daankrol/pi05-snapflow-libero/tree/main/results), [benchmark script](https://huggingface.co/Daankrol/pi05-snapflow-libero/blob/main/benchmark/benchmark_snapflow_libero.py), resolved training config, and Gemma license notice. The evaluated 20k checkpoint SHA-256 is `eec7419542f4b0cb5e2ab57948eb5e03c7ccad141db4d679c6716ee8a050322b`.

To rerun an evaluation, accept the Gemma terms on Hugging Face, install the CUDA, Pi0.5, and LIBERO extras in a Studio `library/` checkout, then download the checkpoint and script from the Hub repository. The result JSON records the teacher and dataset revisions.
