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

The **AWS Provider** training target runs one-off training jobs on AWS Batch.

For cloud-based remote training, the Hugging Face token may be exposed in logs depending on your cloud provider settings. We recommend using short-lived tokens scoped to the access required for training.

The supplied stack scales compute to zero when idle. Keep it deployed for future training jobs; storage charges still apply.

### Deploy the stack

1. In Studio, open **Settings**, then **Training Targets**, and select **AWS Provider**.
2. Select **Deploy AWS Batch stack** to open the CloudFormation template in the AWS Console, or use an existing deployment. Switch the AWS Console to a region supported by the template.
3. Select the IAM acknowledgement checkbox and create the stack.
4. Wait until the stack status is `CREATE_COMPLETE`, then copy `StackIdentifier` (`<account-id>-<region>-<stack-uuid>`) from the stack's **Outputs** tab.

### Provision robot identity

A robot identity gives the robot its own machine identity so Studio can authenticate with external services such as AWS.

> [!NOTE]
> Using host AWS credentials (`~/.aws` profiles) is intended for development only and is not recommended in production. To use a host profile during development, skip to [Register the training target](#register-the-training-target).

Before provisioning, ensure:

- the administrator machine has AWS CLI v2 authenticated for the stack's account, `curl`, and `ssh`/`scp`;
- the Studio host has `snapd`, `sudo` access, and outbound HTTPS (443) connectivity to AWS IoT endpoints.

From the administrator machine, run [`provision-robot.sh`](../cloudformation/provision-robot.sh) with the `StackIdentifier` output from the stack:

```bash
application/cloudformation/provision-robot.sh \
  --thing-name robot-01 \
  --stack-identifier 123456789012-us-east-1-6ba57d60 \
  --ssh <user>@<studio-host>
```

The script creates an AWS IoT thing and active X.509 certificate, delivers the bundle to the robot/Studio host over SSH, installs the `aws-iot-greengrass` snap, and writes `/var/snap/aws-iot-greengrass/common/greengrass/v2/config.yaml` (override with `GREENGRASS_CONFIG_PATH` if stored at a custom path). When this configuration file is present on the Studio backend host, Studio exchanges the device certificate for temporary AWS credentials automatically.

Without SSH access from the administrator machine, omit `--ssh` to write the bundle to `./robot-bundles/<thing-name>`, copy that directory to the Studio host, and run `sudo ./install-greengrass-snap.sh <bundle-dir>` there.

> [!WARNING]
> A copy of the private key (`private.pem.key`) remains in the local bundle directory on the administrator machine. Delete the local bundle directory once installation completes.

To prevent a robot from obtaining new temporary AWS credentials without removing its installation, deactivate its certificate using the certificate ID printed during provisioning. Credentials already issued remain valid until they expire (up to `RobotCredentialDurationSeconds`, which can be 12 hours); certificate deactivation alone does not immediately revoke existing role sessions:

```bash
aws iot update-certificate --region <region> --certificate-id <certificate-id> --new-status INACTIVE
```

To remove a robot identity and uninstall the Greengrass snap completely, run [`deprovision-robot.sh`](../cloudformation/deprovision-robot.sh) from the administrator machine. Deprovision all robots before deleting the CloudFormation stack so no certificates remain attached to the stack's IoT policy:

```bash
application/cloudformation/deprovision-robot.sh \
  --thing-name robot-01 \
  --stack-identifier 123456789012-us-east-1-6ba57d60 \
  --ssh <user>@<studio-host>
```

### Register the training target

Return to the **Deploy and configure AWS stack** dialog in Studio (**Settings > Training Targets > AWS Provider**):

1. Enter a **Name** for the training target.
2. Paste `StackIdentifier` into **AWS stack ID**.
3. Connect to the stack:
   - **With a provisioned robot identity**: Select **Connect**. Studio exchanges the device certificate for temporary credentials, verifies the stack's AWS Batch job queues, and saves the target.
   - **With a host AWS profile (development only)**: Select **Next**, choose an **AWS profile** that is configured and authenticated on the Studio backend host, and select **Connect**. If validation fails, an error toast explains recovery and the form preserves your inputs. **Back** returns to the stack ID step without discarding your profile selection.

Editing a target uses the same dialog prefilled with the saved stack ID (and profile, when used), without the deployment link. Select **Save changes** to validate and save it.

### Host AWS profiles (development only)

> [!WARNING]
> Authenticating with a named AWS profile on the Studio backend host is for development only and is not recommended in production. For production deployments, [provision a robot identity](#provision-robot-identity) instead.

When no robot certificate is configured on the host, Studio lists named profiles from `~/.aws` on the backend host. Configure and authenticate a profile with permission to assume the stack's Studio role (for example, using `aws login` or `aws configure --profile studio`), then verify it:

```bash
aws sts get-caller-identity --profile studio
```

In Docker deployments, mount `~/.aws` so it is readable by the backend container user.

Select the profile in Studio when connecting the target (reopen the dropdown to refresh after setup). If the profile's session expires later, renew it on the backend host and select **Check status** in Studio; running jobs continue on AWS and resume reporting in Studio once credentials are renewed.

### Train with AWS Batch

Select **AWS Provider** in the training dialog and choose a policy. Studio selects the instance type, queue, and job definition automatically:

| Policy       | Instance type  |
| ------------ | -------------- |
| ACT, SmolVLA | `g4dn.2xlarge` |
| Pi0.5        | `g6e.2xlarge`  |
| MolmoAct2    | `g7e.8xlarge`  |

Studio uploads the dataset snapshot to S3, submits the Batch job, streams logs and live metrics while it runs, and downloads the trained model when it finishes. Canceling a job in Studio terminates the AWS Batch job.
