<p align="center">
  <img src="https://raw.githubusercontent.com/open-edge-platform/physical-ai-studio/main/docs/assets/physical_ai_studio.png" alt="Physical AI Studio Application" width="100%">
</p>

# Physical AI Studio Application

Studio application for collecting demonstration data, managing datasets, training VLA model policies, and running trained policies on robot environments.

The application provides a graphical interface to:

- Set up robot arms and cameras.
- Create reusable robot-camera environments.
- Record and review demonstration datasets.
- Train policies using the PhysicalAI library.
- Run trained policies in Studio or deploy them with [OpenVINO PhysicalAI](https://github.com/openvinotoolkit/physicalai).

<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="https://raw.githubusercontent.com/open-edge-platform/physical-ai-studio/main/docs/assets/application.gif" alt="Application demo" width="100%">
</p>
<!-- markdownlint-enable MD033 -->

## Start Here

| Task                        | Documentation                                                             |
|-----------------------------|---------------------------------------------------------------------------|
| Install the application     | [Installation](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/01-installation.md)                                 |
| Update an existing setup    | [Update Existing Installation](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/02-update-existing-installation.md) |
| Complete the first workflow | [Getting Started](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/03-getting-started.md)                           |

## Application Guides

| Guide                                                                     | Description                                                                       |
|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [Installation](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/01-installation.md)                                 | Install with Docker or run backend and UI natively.                               |
| [Update Existing Installation](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/02-update-existing-installation.md) | Refresh Docker images, dependencies, and services after pulling changes.          |
| [Getting Started](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/03-getting-started.md)                           | Create a project, set up hardware, record data, train a model, and run inference. |
| [Environment Setup](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/04-environment-setup.md)                       | Configure robots, cameras, and environments.                                      |
| [Virtual USB Ports](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/08-virtual-usb-ports.md) (optional)            | Connect to robot serial devices over TCP using `socat` virtual ports.             |
| [Recording Datasets](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/05-recording-datasets.md)                     | Record, review, import, and export demonstration datasets.                        |
| [Training Policies](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/06-training-policies.md)                       | Train model policies from recorded datasets.                                      |
| [Deploying Model Policies](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/07-deploying-model-policies.md)         | Run trained policies in Studio or deploy them externally.                         |

## Components

| Component                 | Description                                                    | Documentation                         |
|---------------------------|----------------------------------------------------------------|---------------------------------------|
| **[Backend](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/backend/)** | FastAPI server for data management and training orchestration. | [Backend README](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/backend/README.md) |
| **[UI](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/ui/)**           | React web application.                                         | [UI README](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/ui/README.md)           |
| **[Docker](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docker/)**   | Containerized application runtime.                             | [Docker README](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docker/README.md)   |

## See Also

- [Main Repository](https://github.com/open-edge-platform/physical-ai-studio/blob/main/README.md) - Project overview and library quick start.
- [Library](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/README.md) - Python SDK for programmatic usage.
