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
3. Select **Deploy AWS stack**.
4. Set **Policy to train** to the policy that will run on this trainer.
5. Paste the SSH public key corresponding to the private key on the Studio host.
6. Click apply and wait until the stack status is `CREATE_COMPLETE`.

Stack creation can take up to 5-10 minutes.

### Register the trainer

Open the completed stack's **Outputs** tab, then return to the **Add training target** dialog in Studio.

1. Enter a name for the trainer.
2. Select **Connection details**.
3. Copy the stack outputs into the form:

| CloudFormation output | Studio field    |
| --------------------- | --------------- |
| `SshHost`             | **Host**        |
| `SshPort`             | **Port**        |
| `SshUserName`         | **User**        |

4. Set **Key path** to the private key file corresponding to the public key passed to CloudFormation. The path is resolved on the Studio backend host.
5. Select **Add training target**.

After adding the trainer, return to [Training Policies](./06-training-policies.md) to create and start a model training job.

## AWS Batch provider

The **AWS Provider** training target submits one-off training jobs to AWS Batch. Studio supports one AWS provider with the fixed name **AWS Provider**. Studio uses the AWS SDK credential chain on its backend host to assume the configured Studio role.

1. Open **Settings**, then **Training Targets**, and select **AWS Provider**.
2. Deploy the linked AWS Batch CloudFormation stack, or provision equivalent resources with Terraform or another tool.
3. Enter the stack's `ConfigurationUri` output in **S3 configuration file URI**.
4. Select **Add training target**.

The AWS Batch template publishes `studio-config.json` in its jobs bucket. Studio reads and validates this document with the backend's existing AWS credentials before saving the training target. That identity needs `s3:GetObject` permission on the configuration object and permission to assume the role specified in the document. The role must trust that identity. Cross-account access also requires the bucket policy to allow the identity to read the configuration object.

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
			"job_definition": "my-training-definition:1"
		}
	}
}
```

`bucket` identifies the bucket for datasets, progress, and model artifacts. Each `targets` entry associates an instance type with its Batch queue and job definition. The provisioned compute environment and job definition must match that entry. At least one target is required; documents are limited to 64 KiB and schema version 1.

Studio stores the resolved resource configuration with the training target. After provisioning changes, open **Edit** and select **Save changes** to reload the S3 document. Existing AWS Batch registrations without a configuration URI require one when edited. An upgrade rejects databases containing multiple AWS providers; remove the extra registrations before upgrading.

## Remove an AWS trainer

Delete the CloudFormation stack when training is complete. Removing the trainer from Studio does not delete the AWS resources or stop AWS charges.
