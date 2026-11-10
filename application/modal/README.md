# Train on Modal

Studio deploys and connects a new NVIDIA trainer on Modal. Existing saved connections remain editable. It uploads a dataset snapshot to a Modal Volume, starts one asynchronous GPU call, streams progress and metrics, and downloads the exported model into Studio. No public trainer HTTP endpoint, SSH host, or AWS resources are needed.

## Deploy from Studio

1. Configure Modal credentials for the **Studio backend process**, using `modal setup` as its OS user or `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. A login on your laptop does not authenticate a backend running in Docker or on another host. Use your secret manager or a Compose override; never bake credentials into an image.
2. In **Settings → Training Targets → Modal Provider**, enter a name. Choose the NVIDIA GPU later when starting each training job. Under **Advanced settings**, optionally choose an existing Modal environment and a named Hugging Face Secret as a fallback when Studio has no configured token. Credentials and secret contents are never entered into the form.
3. Review the cloud-cost notice and select **Deploy and connect**. This action confirms creation of a new app and storage Volume; there is no separate setup mode or confirmation checkbox. The first image build can take several minutes. Studio saves the target first, then reports deployment progress and errors through its health status. You can close the dialog without stopping deployment.

Studio generates unique app and Volume names and refuses to deploy if either resource already exists. It deploys the bundled trainer with the CUDA image and the resource defaults listed below. GPU selection is saved with each training job, not the target or deployment request. Existing apps are not updated by this flow. Editing a target only edits its connection; it does not redeploy the function.

Deployment can incur image-build and storage charges; later training incurs GPU charges. Deleting a target attempts to cancel its training calls, stop its app, and permanently delete its Volume, even if a deployment is still running. Each cleanup step is bounded and attempted independently; the local target is removed even if cloud cleanup fails. Check the Modal dashboard for remaining resources and charges after an error. Active deployment targets cannot be edited, but can be deleted.

Deployment progress is process-local, like managed SSH setup. Restarting Studio interrupts the local deployment command, but cloud work already accepted by Modal can continue. After a restart or failure, inspect the saved target and the Modal dashboard before creating another deployment. Partial resources are retained until the target is removed or you clean them up in Modal. A cloud build already accepted before deletion may still finish later; cleanup is best-effort, not proof that all remote resources have stopped.

The automatic flow uses `ghcr.io/open-edge-platform/physicalai-trainer-cuda:main`. Advanced manual deployments can use a custom image or other GPU types, but new connections to them must be registered through the remote-trainer API; the add dialog always creates a new deployment.

## Choose a GPU for a training job

In **Train model**, select the Modal target under **Run on**, then use **NVIDIA GPU** to choose **A10G**, **A100 (80 GB)**, or **H100** for that job. A10G is initially selected; every new job explicitly requests the selected GPU rather than using a deployment default. Studio saves the choice with the job and applies it with Modal's per-invocation options; it does not redeploy or replace the function's image, Volume, or timeout. When a Studio Hugging Face token is configured, its runtime Secret is added while preserving existing named secret bindings. GPU changes affect billing and availability. Reattaching to a running job reconnects to its original call without starting another GPU.

Studio requires Modal SDK 1.6.3 or newer for per-job function options. Its backend dependency includes `api-proxy-support`, which is required when the Modal API is reached through an HTTP/SOCKS proxy. Without it, `modal setup` can stall before opening a browser and eventually report that it could not connect.

## Deploy manually

You need a Modal account with GPU access and permission to deploy apps and manage Volumes. From the repository root:

```bash
uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal setup
uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal deploy --env main application/modal/train.py
```

Defaults:

| Resource | Default |
| --- | --- |
| App / function | `physicalai-trainer` / `train` |
| Environment | `main` |
| Volume | `physicalai-training` |
| Fallback GPU (calls without a per-job selection) | One `A10G` |
| CPU / RAM | 8 cores / 32 GiB |
| Execution timeout | 24 hours per call |
| Container image | `ghcr.io/open-edge-platform/physicalai-trainer-cuda:main` |

The image must contain Studio's `trainer.batch` worker. The deployment script restores pip in the image so Modal can install its container runtime; it does not change the base trainer Dockerfile.

Set these environment variables **before deployment** to customize the app:

```bash
export PHYSICALAI_MODAL_GPU=A100-80GB
export PHYSICALAI_MODAL_APP=physicalai-large-trainer
export PHYSICALAI_MODAL_VOLUME=physicalai-large-training
# Recommended for reproducibility: use a tested trainer image tag or digest.
export PHYSICALAI_MODAL_IMAGE=ghcr.io/open-edge-platform/physicalai-trainer-cuda:main
uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal deploy --env main application/modal/train.py
```

Choose a GPU with sufficient memory for your policy and batch size. All Studio-supported policies are offered without checking the deployed image's dependencies or GPU compatibility. Selecting a policy does **not** automatically select an appropriate GPU; missing dependencies or insufficient GPU memory can cause the job to fail. Studio uses the GPU selected for each training job, regardless of the manual deployment's fallback GPU. For a different environment, change `--env` and use the same environment in Studio.

Studio forwards the Hugging Face token configured in Settings (falling back to the backend's `HF_TOKEN` environment variable) as an ephemeral Modal runtime Secret for each new training call. The token leaves Studio and is available inside your Modal container, but is never included in the persisted job payload, dataset Volume, image, or Studio progress messages. The Studio token takes precedence over a deployment's `HF_TOKEN`; without a Studio token, deployment secrets are unchanged. Reattachment keeps the original running call and credentials.

Alternatively, create a named Modal Secret in the deployment environment with an `HF_TOKEN` entry and set `PHYSICALAI_MODAL_SECRET` to its name before manual deployment. Existing named secrets are preserved when forwarding the Studio token. Custom deployments with unnamed secrets must switch those bindings to named Modal Secrets before using Studio token forwarding; Studio refuses to silently discard them. Keep credentials out of source control, grant access to required gated models, and use short-lived, read-only tokens.

## Existing connections

Existing saved targets remain editable in **Settings → Training Targets**. Editing updates the name, app, function, Volume, or environment without deploying new resources. The remote-trainer API still accepts connections to manually deployed apps; this compatibility path is not a setup choice in the add dialog.

Give the **Studio backend process** Modal credentials for the deployment's workspace. A CLI login on your laptop does not authenticate Docker or another host. Supply credentials to the backend container or host without baking them into an image.

Check connectivity, then select the target in the model training dialog. All policies are enabled without a manual list or capability discovery. Choose the GPU for each training job. Modal always uses NVIDIA CUDA device 0, regardless of Studio's local hardware. The training dialog defaults to BF16 Mixed and offers CUDA batch-size auto-scaling; precision can still be changed.

The health check verifies access to the named function and Volume without allocating a GPU. It does not check the function's implementation, Volume mount, model access, or GPU capacity. The function must mount the **same Volume** configured in Studio at `/training-jobs` and follow this script's job-file contract. Only connect trusted deployments. Backend Modal credentials authorize cloud spend and data access; restrict access to Studio and use a dedicated workspace where appropriate.

## Lifecycle and operations

- **Progress and metrics:** Volume snapshots are polled every five seconds. Full trainer stdout/stderr and build errors are available in the Modal dashboard, not mirrored into Studio's log file.
- **Studio restart:** a persisted Modal function-call ID reattaches to the existing call, without re-uploading the dataset or starting another GPU. A graceful Studio shutdown leaves training running and billable. A crash in the small window between submitting a call and saving its ID can leave an orphan call; inspect the Modal dashboard before retrying.
- **Connectivity:** Studio keeps the saved function-call ID and retries monitoring when Modal connectivity or authentication is unavailable. Training may continue running and billing during the outage. Restore connectivity/credentials or stop the call in the Modal dashboard; Studio does not treat lost contact as confirmed training failure.
- **Cancel:** Studio calls Modal's native cancellation API. If connectivity or authentication fails, cancellation remains pending and is retried rather than reported as completed. A Studio shutdown preserves the pending request for reattachment. Verify termination in Modal if connectivity or credentials fail. Abrupt termination may leave the last progress snapshot stale.
- **Timeouts:** one GPU, one execution attempt, at most 24 hours. Automatic retries, multi-GPU training, checkpoint continuation across calls, and resuming from a Studio base model are not supported.
- **Retention:** `jobs/<studio-job-uuid>/` contains `spec.json`, `dataset.zip`, `status.json`, `metrics.csv` (when produced), and `artifact.zip` (on success). Files persist after success, failure, and cancellation. Removing the Studio target attempts to permanently delete the **entire configured Volume**, including all job data. Successfully imported models stay in Studio, but unfinished or not-yet-imported results can be lost. Do not point multiple targets at the same app or Volume unless you intend their deletion to affect all users. Cleanup does not delete shared named Secrets or registry images. Volume storage is billed separately from GPU compute; check Modal if cleanup fails.
- **Reattachment limits:** native call results expire after seven days. Completed jobs can still be imported from their retained Volume status and artifact; incomplete jobs whose native call has expired cannot be monitored reliably.

Before using real datasets, smoke-test a small ACT job: observe metrics, restart Studio during training, verify artifact import, and cancel a second job while watching its container terminate in Modal. Unit tests use fake SDK resources; they do not prove GPU availability, image compatibility, or account permissions.
