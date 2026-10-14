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

1. In Studio, open **Settings**, then **Compute**.
2. Under **Remote Trainers**, select **New remote trainer**.
3. Select **Deploy AWS stack**.
4. Set **Policy to train** to the policy that will run on this trainer.
5. Paste the SSH public key corresponding to the private key on the Studio host.
6. Click apply and wait until the stack status is `CREATE_COMPLETE`.

Stack creation can take up to 5-10 minutes.

### Register the trainer

Open the completed stack's **Outputs** tab, then return to the **Add remote trainer** dialog in Studio.

1. Enter a name for the trainer.
2. Select **SSH tunnel** and **Connection details**.
3. Copy the stack outputs into the form:

| CloudFormation output | Studio field    |
| --------------------- | --------------- |
| `SshHost`             | **Host**        |
| `SshPort`             | **Port**        |
| `SshUserName`         | **User**        |

4. Set **Key path** to the private key file corresponding to the public key passed to CloudFormation. The path is resolved on the Studio backend host.
6. Select **Add trainer**.

After adding the trainer, return to [Training Policies](./06-training-policies.md) to create and start a model training job.

## Remove an AWS trainer

Delete the CloudFormation stack when training is complete. Removing the trainer from Studio does not delete the AWS resources or stop AWS charges.
