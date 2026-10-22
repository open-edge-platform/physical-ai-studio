# Remote Training

Remote training executes a training job on a GPU-equipped trainer host. Studio uploads a dataset snapshot, monitors the job, and downloads the model artifacts when training finishes.

Use remote training when the Studio backend host does not meet the policy's GPU requirements or when training must run on dedicated infrastructure.

## Prepare an SSH host

If you select **Set up Docker and GPU support** for an Ubuntu SSH trainer, Studio installs missing packages and configures Docker on that host. The SSH user needs passwordless sudo for these changes (and for a separately confirmed reboot); an already-ready host does not need it for the prerequisite check. A root SSH user does not need sudo.

On the **SSH host**, have an administrator grant the dedicated SSH user non-interactive sudo. For example, replace `trainer` with the SSH username you enter in Studio:

```bash
sudo visudo -f /etc/sudoers.d/physicalai-trainer
```

Add this line in the editor and save:

```text
trainer ALL=(ALL) NOPASSWD: ALL
```

Verify the file has permissions `0440` (`sudo chmod 0440 /etc/sudoers.d/physicalai-trainer` if needed), then log in as that SSH user and check `sudo -n true` succeeds without a prompt before selecting setup in Studio. If it fails, ask the host administrator to check the sudoers policy. Do not put the rule on the Studio backend host: installation runs on the trainer host.

> [!WARNING]
> `NOPASSWD: ALL` grants unrestricted root access to that SSH user. Use only a trusted, dedicated account on a host you administer; consult your administrator before enabling it. Docker group membership, which Studio may also add, is root-equivalent. If you cannot grant this access, have an administrator prepare Docker and GPU support manually instead of selecting automatic setup.

## AWS

Studio can create a GPU-backed remote trainer in your AWS account and connect to it through an SSH tunnel.

For cloud-based remote training, the Hugging Face token may be exposed in logs depending on your cloud provider settings. We recommend using short-lived tokens scoped to the access required for training.

> [!WARNING]
> Delete the CloudFormation stack when you finish training to terminate the EC2 instance and avoid additional AWS charges.

### Prerequisites

Before deploying the stack, you need:

- an AWS account with permission to create the stack and its resources;
- an SSH key pair on the Studio host. The CloudFormation form includes instructions to create one;
- enough EC2 On-Demand quota and capacity for the selected GPU instance type.

### Restrictive networks

If the Studio backend must use a proxy for outbound connections, set the standard proxy environment variables before starting Studio:

```bash
HTTPS_PROXY=http://proxy.example.com:8080
HTTP_PROXY=http://proxy.example.com:8080
```

When running Studio with Docker Compose, set these variables in `application/docker/.env` before starting the containers.

### Deploy the stack

1. In Studio, open **Settings**, then **Training Targets**.
2. Under **Add a training target**, select **SSH Remote Trainer**.
3. Select **Deploy AWS stack**. The template requires **US East (N. Virginia)** (`us-east-1`).
4. Set **Policy to train** to the policy that will run on this trainer.
5. Paste the SSH public key corresponding to the private key on the Studio host.
6. Click apply and wait until the stack status is `CREATE_COMPLETE`.

Stack creation can take up to 5-10 minutes.

### Register the trainer

Open the completed stack's **Outputs** tab, then return to the **Add training target** dialog in Studio.

1. Enter a name for the trainer.
2. Select **Connection details**.
3. Copy the stack outputs into the form:

| CloudFormation output | Studio field |
| --------------------- | ------------ |
| `SshHost`             | **Host**     |
| `SshPort`             | **Port**     |
| `SshUserName`         | **User**     |

4. Set **Key path** to the private key file corresponding to the public key passed to CloudFormation. The path is resolved on the Studio backend host.
5. Select **Add training target**.

Studio automatically assigns an available local loopback port for each SSH trainer. After adding the trainer, return to [Training Policies](./06-training-policies.md) to create and start a model training job.

### Remove an EC2 trainer

The EC2 instance keeps running after training finishes. Delete its CloudFormation stack when you finish using the trainer to terminate the instance and stop its compute charges. Removing the trainer from Studio only deletes its registration; the AWS resources remain deployed.

## AWS Batch provider

The **AWS Provider** training target submits one-off training jobs to AWS Batch. Studio supports one AWS provider with the fixed name **AWS Provider**. Studio uses the selected named AWS profile on its backend host to assume the configured Studio role.

For cloud-based remote training, the Hugging Face token may be exposed in logs depending on your cloud provider settings. We recommend using short-lived tokens scoped to the access required for training.

The supplied stack scales compute to zero when idle. Keep it deployed for future training jobs; storage charges still apply.

You can train multiple models at the same time. When deploying the AWS stack, increase **Parallel trainers per EC2 pool** (1–16, default 1) to allow more training jobs to run simultaneously. Jobs beyond the configured capacity wait automatically until a running job finishes.

1. Open **Settings**, then **Training Targets**, and select **AWS Provider**.
2. In **Deploy and configure AWS stack**, deploy the linked CloudFormation stack in **US East (N. Virginia)** (`us-east-1`), or use an existing deployment. Wait for stack creation to complete and copy `ConfigurationUri` from its **Outputs** tab.
3. Paste that output into **S3 configuration file URI**, then select **Next**. Use the configuration file URI, not just the bucket URL.
4. In **Connect to AWS remote training stack**, select an **AWS profile** that is configured and authenticated on the Studio backend host. The information button explains credential setup.
5. Select **Connect**. Studio reads and validates the configuration using the selected profile, then saves the target. If it fails, an error toast explains recovery and the form preserves your inputs. **Back** returns to the URI without discarding your profile selection.

Editing a target uses the same two steps, prefilled with the saved URI and profile, without the deployment instructions. Select **Save changes** on the second step to validate and save it.

The AWS Batch template publishes `studio-config.json` in its jobs bucket. Studio reads and validates this document with the backend's existing AWS credentials before saving the training target. That identity needs `s3:GetObject` permission on the configuration object and permission to assume the role specified in the document. The role must trust that identity. Cross-account access also requires the bucket policy to allow the identity to read the configuration object.

### Authenticate the Studio backend

The **AWS profile** dropdown lists profiles configured on the backend host. Select a profile explicitly; there is no ambient credential-chain option. A profile named `default` can be selected like any other profile. SSO, assume-role profiles, and configured credential helpers are supported. Studio saves the profile name with the target and pins it in submitted jobs for restart recovery; it does not copy credentials into the database or job payload. The training region comes from the configuration document. Open the dropdown again to refresh profiles after external setup.

On a fresh machine, install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) if your authentication method uses it, then create the profile on the **backend host**. For organization-managed IAM Identity Center access, your administrator supplies the SSO start URL and SSO region:

```bash
aws configure sso --profile studio
aws sso login --profile studio
aws sts get-caller-identity --profile studio
```

For IAM access keys, use `aws configure --profile studio` instead. Other credential helpers should be configured according to your organization's instructions. Select `studio` in Studio's profile dropdown; subsequent starts do not require `AWS_PROFILE` or region exports. Renew authentication using the profile's usual method when its session expires.

Submitting the form verifies that the selected profile can read the S3 configuration; it does not verify permissions for every training resource. In Docker deployments, make `~/.aws` configuration, credentials, and session caches accessible to the backend container user, and persist them across container recreation. Studio reads profiles through the AWS SDK and does not require the AWS CLI to be installed in the container.

If a saved target's session expires, **Health & capability** shows **Credentials expired**. Renew the profile using its configured authentication method on the backend host (for example, `aws sso login --profile studio`), then select **Check status** in Studio.

Studio also reports authentication issues when launching training and while tracking jobs, logs, metrics, or model downloads. The error toast names the affected profile and offers **Copy command** for SSO renewal. Copying dismisses the toast. Job actions and target health provide **Authentication instructions** to show it again.

An already-submitted AWS job is not marked failed just because Studio's credentials expire. Studio retains its job ID and last progress, shows **Awaiting AWS authentication**, and retries automatically after renewal. AWS training may continue while Studio cannot observe it. Model downloads and live metrics resume after credentials are renewed. If stopping requires authentication, **Stop pending — authentication required** means AWS termination has not yet been confirmed; the cancellation request survives a Studio restart. Network and permission failures are reported separately from authentication issues.

### Configuration document

Terraform and other provisioning tools can publish the same JSON document to an S3 object:

```json
{
  "schema_version": 1,
  "region": "eu-west-1",
  "studio_role_arn": "arn:aws:iam::123456789012:role/studio",
  "bucket": "my-training-jobs",
  "targets": {
    "g4dn.xlarge": {
      "queue": "my-training-queue",
      "job_definition": "my-training-definition:1",
      "max_trainers": 4
    }
  }
}
```

`bucket` identifies the bucket for datasets, progress, and model artifacts. Each `targets` entry associates an instance type with its Batch queue and job definition. The provisioned compute environment and job definition must match that entry. At least one target is required; documents are limited to 64 KiB and schema version 1.

The optional positive integer `max_trainers` reports the pool's configured maximum concurrent trainers for display; AWS enforces capacity through the compute environment. Older configurations without it display **Not reported**. Each training job stores its inputs and outputs under its own `jobs/<studio-job-id>/` S3 prefix, isolating parallel runs.

Studio stores the resolved resource configuration with the training target. After provisioning changes, open **Edit** and select **Save changes** to reload the S3 document. Existing AWS Batch registrations without a configuration URI require one when edited. An upgrade rejects databases containing multiple AWS providers; remove the extra registrations before upgrading.

### Train with AWS Batch

Select **AWS Provider** in the training dialog and choose a policy. Studio selects its instance type, queue, and job definition automatically:

| Policy       | Instance type  |
| ------------ | -------------- |
| ACT, SmolVLA | `g4dn.2xlarge` |
| Pi0.5        | `g6e.2xlarge`  |
| MolmoAct2    | `g7e.8xlarge`  |

Only policies with a provisioned target appear. AWS Batch targets use the declared capabilities of the CUDA trainer image; they do not expose an HTTP policy-discovery endpoint or require a GPU selection.

Studio uploads the dataset snapshot and training specification to S3, submits a Batch job, monitors Batch state and S3 progress, and downloads the completed model. Canceling training terminates the Batch job. Restarting Studio reattaches to submitted jobs using their persisted Batch job IDs and resource configuration.

Container stdout and stderr are collected in CloudWatch Logs and relayed into Studio's job logs. Live metric snapshots are published to S3 and streamed through Studio's metrics endpoint. Existing deployments need a CloudFormation stack update to grant the Studio role `logs:GetLogEvents` on the trainer log group, and a rebuilt trainer image to publish live metrics. A job remains running while AWS Batch reports it running, unless the trainer explicitly reports failure; a traceback alone does not imply that the container exited.

Studio refreshes the Batch lifecycle message every five seconds, including elapsed time, scheduling reasons, and the age of the latest trainer report. Batch-level training updates without a message display **Training model** rather than retaining an earlier **Dataset ready** message. During startup, ECS task diagnostics distinguish image pulling from container setup when AWS reports those timestamps; update existing stacks to grant the Studio role `ecs:DescribeTasks` for this detail. Without that permission, Studio still reports the Batch state. AWS does not expose an image download percentage.
