# Hugging Face Integration

Several policies download assets from Hugging Face Hub (for example, SmolVLA,
Pi0.5, and other Hub-backed models).

If no Hugging Face token is configured, the backend logs a warning and Hub
access is unauthenticated.

Configure a token for any workflow that depends on Hugging Face-hosted assets.
Without a token, model downloads may fail (for example, due to anonymous rate
limits or access restrictions on gated/private repositories).

## Required token permissions

For model download in Physical AI Studio, use a token with **read-only** Hub
access:

- **Required:** `Read` permission for model repositories.
- **Not required:** `Write` or admin permissions.
- **If using gated/private models:** the token must belong to a Hugging Face
  account that has been granted access to those repositories.

If you use a fine-grained token, grant read access to the specific model repos
you plan to train from.

## Create a Hugging Face token

1. Sign in to [huggingface.co](https://huggingface.co/).
2. Open **Settings** -> **Access Tokens**.
3. Create a new token.
4. Set permissions to read-only model access (see required permissions above).
5. Copy the token value.

## Configure the token

### Native backend

Set the token from the UI: **Settings > General > Hugging Face**. This persists
the token to the backend's `settings.json` file.

Alternatively, edit `settings.json` directly (see
[`get_settings_file_path`](../src/settings.py) for its location) and set
`huggingface.hf_token`.

### Docker deployment

Add the token to `application/docker/.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

docker-compose injects this as a real container environment variable. Then run
Docker Compose as usual:

```bash
cd application/docker
docker compose up
```

## Verify setup

- Start a training job for a Hub-backed policy (for example, SmolVLA or Pi0.5).
- Confirm there is no warning about a missing Hugging Face token.

## Security notes

- Never commit real tokens to source control.
- Store tokens in `settings.json`, the Docker `.env` file, or your secret manager.
- Rotate the token immediately if it is exposed.
