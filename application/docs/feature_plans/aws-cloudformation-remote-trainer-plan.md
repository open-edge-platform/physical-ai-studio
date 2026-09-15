# AWS CloudFormation Remote Trainer Integration Plan

## Objective

Provide a low-friction way for a user to provision a Physical AI Studio remote
trainer on AWS by following a CloudFormation quick-create link.

CloudFormation owns the AWS infrastructure and the long-running trainer
container. An SSH tunnel makes the remote trainer API available as a local URL
on the host running the Studio backend. Studio then uses its existing direct
remote-training flow unchanged.

## Architecture

```text
Host running Studio backend
    |
    | http://127.0.0.1:8001
    v
Manual SSH port-forward
    |
    v
AWS EC2: 127.0.0.1:8001
    |
    v
Long-running Physical AI trainer container
```

### Ownership Boundaries

CloudFormation owns:

- VPC and subnet
- Internet gateway and routing
- Security group
- EC2 instance
- SSH public-key installation
- Docker installation
- GPU runtime configuration through a tested GPU-ready AMI
- Encrypted EBS storage
- Long-running trainer container
- Container restart policy
- Infrastructure and trainer readiness checks

Studio owns:

- Direct remote-trainer registration
- Training job submission
- Dataset snapshot creation and packaging
- Dataset upload through the trainer HTTP API
- Training status and progress polling
- Trained model archive download
- Importing the downloaded model into local Studio storage

The user owns:

- AWS stack creation and deletion
- The SSH private key
- Starting and stopping the SSH tunnel
- EC2 stop/start for cost control
- Registering the forwarded trainer URL in Studio

Studio must not:

- Provision or modify the EC2 instance
- Access the remote Docker daemon
- Start or stop the remote trainer container
- Receive or store the SSH private key
- Know that the trainer URL is backed by an SSH tunnel

## Relationship to Existing Features

This integration uses the existing **direct remote-trainer** functionality.

The registered trainer URL is:

```text
http://127.0.0.1:8001
```

The SSH tunnel forwards that local endpoint to the trainer container running
on AWS.

The existing Studio-managed **SSH-provisioned trainer** feature is separate
and must remain unchanged. These are parallel features. Do not remove,
deprecate, modify, or merge the SSH-provisioned path as part of this work.

No Studio backend or UI changes are planned for the initial CloudFormation
integration.

## Dataset and Model Transfer

Existing direct remote-training APIs handle all data transfer through the SSH
tunnel:

1. Studio creates a snapshot of the selected dataset.
2. Studio packages the snapshot as an archive.
3. Studio submits a remote job through the trainer HTTP API.
4. Studio streams the dataset archive through the SSH tunnel.
5. The trainer extracts the dataset and trains remotely.
6. Studio polls the trainer HTTP API for status and progress.
7. The trainer packages the trained model and weights.
8. Studio downloads the model archive through the same SSH tunnel.
9. Studio imports the model into its local storage.
10. The trainer removes temporary job data according to its existing lifecycle.

Do not introduce S3 dataset transfer, shared filesystems, `scp`, `rsync`, or a
new artifact-transfer protocol.

## Production CloudFormation Template

### Networking

The template must create:

- A dedicated VPC
- A public subnet
- An internet gateway
- A route table with outbound internet access
- A security group exposing only SSH port 22
- An EC2 instance with a dynamic public IP

The trainer HTTP port must not be exposed by the security group.

The container must publish its API only on the instance loopback address:

```text
127.0.0.1:8001
```

### SSH Source Restriction

The user must provide the public CIDR of the host or network running the Studio
backend.

Requirements:

- No permissive default
- Reject `0.0.0.0/0`
- Accept narrow addresses such as `203.0.113.10/32`
- Clearly state that this is the source address from which the Studio backend
  host reaches AWS

### SSH Key

The stack must ask the user to paste one OpenSSH public key, such as:

```text
ssh-ed25519 AAAAC3... user@example
```

Requirements:

- Import the public key as an EC2 key pair
- Never request or accept a private key
- The matching private key must remain on the Linux Studio backend host
- Support standard SSH first-connection host-key confirmation

### EC2 Configuration

The template must:

- Resolve a tested, region-appropriate GPU-ready AMI automatically
- Use one tested GPU instance type as the default
- Allow an advanced instance-type override
- Make no claims that the default instance can train every policy or dataset
- Avoid encoding policy names or policy-specific capacity requirements
- Use encrypted EBS storage
- Delete the EBS storage when the stack is deleted
- Use a dynamic public IP rather than an Elastic IP

The GPU-ready AMI should already provide compatible NVIDIA drivers and NVIDIA
Container Toolkit. Avoid installing the complete GPU driver stack during EC2
bootstrap.

### Trainer Container

The template must:

- Pin a tested immutable trainer image tag or digest
- Avoid exposing image selection to ordinary users
- Start one long-running trainer container
- Enable an appropriate restart policy
- Pass through the GPU
- Mount persistent trainer storage
- Bind trainer port 8001 only to EC2 loopback
- Apply the existing container hardening settings where compatible
- Never mount the Docker socket into the trainer container

### Hugging Face Token

The stack form may accept an optional Hugging Face token:

- Parameter type: string
- Mark it `NoEcho`
- Leave it blank for models that do not require authenticated access
- Pass it to the trainer container
- Recommend a read-only, narrowly scoped token

Document the accepted risk:

- `NoEcho` masks normal CloudFormation displays but does not create a complete
  secret-management boundary.
- Administrators with sufficient CloudFormation, EC2, operating-system, or
  Docker access may recover the token.
- The token must not be committed to the repository.
- Secrets Manager integration is deferred to a later iteration to minimize
  user interaction and stack complexity.

Bootstrap commands must not echo the token or enable shell tracing around
secret handling.

### Readiness

CloudFormation must report successful completion only after:

- EC2 bootstrap has completed
- Docker is running
- The GPU is visible to Docker
- The trainer container is running
- The container health check passes
- The trainer API responds successfully on the EC2 loopback address

A failed bootstrap must produce an actionable CloudFormation failure rather
than a nominally successful but unusable stack.

Do not expose the Hugging Face token in readiness errors, bootstrap logs, or
CloudFormation outputs.

## CloudFormation Outputs

### Tunnel Command

A complete command that the user runs on the host running the Studio backend:

```bash
ssh \
  -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8001:127.0.0.1:8001 \
  <ec2-user>@<instance-public-hostname>
```

The command should run in the foreground for the first iteration. The user
stops it with `Ctrl+C`.

Do not require a `~/.ssh/config` entry in the ordinary path.

### Studio Trainer URL

```text
http://127.0.0.1:8001
```

### Health-Check Command

```bash
curl --fail http://127.0.0.1:8001/health
```

### Troubleshooting Details

Also output:

- EC2 instance ID
- Public hostname or public IP
- Security-group ID
- AWS region

## User Workflow

1. Open the published CloudFormation quick-create link.
2. Enter:
  - The Studio backend host's SSH public key
   - The permitted SSH source CIDR
   - An optional Hugging Face token
3. Accept any required CloudFormation capability acknowledgement.
4. Create the stack.
5. Wait until stack creation completes successfully.
6. Copy the tunnel command from the stack outputs.
7. Run the command on the host running the Studio backend.
8. Confirm the SSH host key on first connection.
9. Leave the SSH command running.
10. Run the output health-check command on that host.
11. Open Studio's existing Compute settings.
12. Register one direct remote trainer with:

    ```text
    http://127.0.0.1:8001
    ```

13. Use the existing Studio training workflow.

The tunnel must run on the Studio backend host because the backend uploads
datasets and downloads model archives.

## Cost-Control Workflow

For the first iteration:

- Users manually stop the EC2 instance when it is not needed.
- Users manually start it before training.
- Automatic idle shutdown is out of scope.
- Because the instance uses a dynamic public IP, stop/start may change its
  address.
- After restart, the user retrieves the updated tunnel command or public
  address from the stack/instance information and starts a new tunnel.
- The trainer's EBS storage remains until the stack is deleted.
- Deleting the stack deletes the encrypted EBS volume and its contents.

Document that users must not stop the instance or delete the stack during an
active training job.

## Development-Only Test Infrastructure

Create a separate temporary CloudFormation template for development.

It must:

- Use an inexpensive non-GPU EC2 instance
- Deploy the real trainer image
- Start the container without GPU passthrough
- Skip only GPU readiness checks
- Bind the trainer API to remote loopback
- Exercise public-key SSH access
- Exercise manual port forwarding
- Exercise trainer health through `http://127.0.0.1:8001`
- Exercise registration through Studio's existing direct-trainer UI

Testing stops after registration and health inspection. Do not submit an actual
training job to this host.

This template is temporary scaffolding:

- Do not link it from customer documentation.
- Do not add a customer-visible test mode to the production template.
- Do not modify Studio validation to accommodate it.
- Remove the temporary template before the feature is delivered.

## Template Publication

For the first iteration, publication is manual:

1. Validate the production template locally.
2. Upload it to a chosen S3 bucket over HTTPS.
3. Ensure CloudFormation can read the object.
4. Construct a quick-create URL:

   ```text
   https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?stackName=physicalai-trainer&templateURL=<URL_ENCODED_S3_TEMPLATE_URL>
   ```

5. Open and test the resulting link.
6. Confirm that the form displays the intended parameters and defaults.
7. Preserve the working launch link as a project deliverable.

CI-based publication is deferred.

## Validation Plan

### Static Validation

- Run `cfn-lint`.
- Run CloudFormation template validation through AWS CLI.
- Validate parameter constraints.
- Confirm `0.0.0.0/0` is rejected.
- Confirm no inbound rule exposes port 8001.
- Confirm no private-key parameter exists.
- Confirm no secret appears in outputs.
- Confirm the trainer image is pinned immutably.

### Development Stack Validation

- Create the inexpensive development stack.
- Confirm the stack reaches its intended readiness state without GPU checks.
- SSH from the Linux Studio backend host.
- Confirm standard host-key verification.
- Start the output tunnel command.
- Confirm local port-forward failure is reported when port 8001 is occupied.
- Call `/health`, `/devices`, and `/storage` through the tunnel.
- Register `http://127.0.0.1:8001` in Studio.
- Confirm Studio displays trainer health.
- Do not submit a training job.

### Production Stack Validation

- Create the production stack with a GPU instance.
- Confirm Docker sees the accelerator.
- Confirm the real trainer container sees the accelerator.
- Confirm the trainer health and device endpoints respond.
- Start the tunnel on the Linux Studio backend host.
- Register the forwarded URL in Studio.
- Submit a small real training job.
- Verify dataset upload through the tunnel.
- Verify status and progress polling.
- Verify model and weight download.
- Verify Studio imports the downloaded model.
- Verify remote temporary data cleanup.
- Stop/start the EC2 instance and verify the documented reconnection flow.
- Delete the stack and confirm infrastructure and EBS cleanup.

## Explicitly Out of Scope

- Changes to Studio-managed SSH provisioning
- New Studio dataset-transfer mechanisms
- S3-based dataset or model exchange
- Shared network filesystems
- Automatic tunnel management
- `systemd`, `autossh`, or another persistent local tunnel service
- Docker-based Studio backend support
- Multiple simultaneous trainers or tunnels
- Elastic IP allocation
- Automatic idle shutdown
- Secrets Manager integration
- CI-based template publication
- Policy-specific EC2 sizing
- Customer-visible networking-test mode

## Agreed Constraints

- One remote AWS trainer per Studio installation in the first iteration
- Native Linux Studio backend only
- Manual foreground SSH tunnel
- Fixed local trainer URL: `http://127.0.0.1:8001`
- Dedicated VPC
- SSH is the only exposed inbound service
- Global SSH CIDR is rejected
- Public key is pasted into the CloudFormation form
- Dynamic public IP
- Manually managed EC2 stop/start
- Template-pinned trainer image
- Optional `NoEcho` Hugging Face token
- Encrypted trainer storage deleted with the stack
- Existing direct remote-training code performs dataset and model transfer
- Existing SSH-provisioned training remains a separate, unchanged feature
- No implementation begins until this plan is explicitly approved for
  execution
