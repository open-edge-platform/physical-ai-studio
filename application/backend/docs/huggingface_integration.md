# Hugging Face Integration

Several policies download assets from Hugging Face Hub (for example, SmolVLA, Pi0,
and other Hub-backed models). Remote training uses direct HTTP dataset transfer by
default; it uses the Hub only when you explicitly set
`TRAINER_DATASET_TRANSFER=hf`.

If `HF_TOKEN` is not set, the backend logs a warning and Hub access is
unauthenticated.

Set `HF_TOKEN` for any workflow that depends on Hugging Face-hosted assets.
Without a token, model downloads may fail (for example, due to anonymous rate
limits or access restrictions on gated/private repositories).

## Required token permissions

The access level depends on the workflow. Each component reads its token from the
environment only.

| Component / workflow | `HF_TOKEN` access | Why |
|----------------------|-------------------|-----|
| Studio backend — local training, recording, deployment | **Read** | Download Hub-backed policy weights (SmolVLA, Pi0, …). |
| Studio backend — remote training with `TRAINER_DATASET_TRANSFER=http` (default) | Not required for transfer | The backend streams the dataset directly to the trainer. A token is still needed for Hub-backed policy assets. |
| Studio backend — remote training with `TRAINER_DATASET_TRANSFER=hf` | **Write** | Create, upload to, and delete a temporary private dataset repo per job under your namespace. |
| Trainer service with `TRAINER_DATASET_TRANSFER=hf` | **Read** | Pull the dataset snapshot from the temporary repo at its pinned commit. |

Notes:

- A token with **write** access also covers read, so a single write token on the
  backend works for Hub-backed policy downloads and optional HF transfer. Prefer
  the least privilege the workflow needs.
- **Gated/private models:** the token's account must have been granted access to
  those repositories.
- **Delete:** remote mode with `TRAINER_DATASET_TRANSFER=hf` deletes each
  temporary repo after import (including on failure), so the backend token must
  be able to delete repos it created.

### Classic (read/write) tokens

- **Read** workflows: create a token with the `Read` role.
- **Remote backend (write)**: create a token with the `Write` role. It can create,
  write, and delete repos under your namespace.

### Fine-grained tokens

If you use a fine-grained token, grant:

- **Read** (downloads, trainer): `Read access to contents of all repos you can
  access`, or scope it to the specific model repos you train from.
- **Write** (remote backend): under **Repositories**, enable `Write access to
  contents and settings of all repos` (or the specific namespace), which allows
  creating, uploading to, and deleting the temporary dataset repos.

## Create a Hugging Face token

1. Sign in to [huggingface.co](https://huggingface.co/).
2. Open **Settings** -> **Access Tokens**.
3. Create a new token.
4. Set permissions for your workflow (see required permissions above): read-only
   for downloads/trainer, write for the remote backend.
5. Copy the token value.

## Configure `HF_TOKEN`

Set `HF_TOKEN` in the environment used by the backend.

### Native backend

Add the token to `application/backend/.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Then start the backend as usual:

```bash
cd application/backend
./run.sh
```

### Docker deployment

Add the token to `application/docker/.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Then run Docker Compose as usual:

```bash
cd application/docker
docker compose up
```

### Trainer service (optional HF dataset transfer)

For `TRAINER_DATASET_TRANSFER=hf`, the trainer service needs its own `HF_TOKEN`
with **read** access. The default `http` transfer does not require a trainer
token unless the selected policy downloads Hub-hosted assets. Add it to
`application/trainer/.env` on the GPU machine when needed:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

See [Remote Training Server](../../docs/08-remote-training-server.md) for the full
setup.

## Verify setup

- Start a training job for a Hub-backed policy (for example, SmolVLA or Pi0).
- Confirm there is no warning about missing `HF_TOKEN`.

## Security notes

- Never commit real tokens to source control.
- Store tokens in local `.env` files or your secret manager.
- Rotate the token immediately if it is exposed.
