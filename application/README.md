<p align="center">
  <img src="../docs/assets/physical_ai_studio.png" alt="Physical AI Studio Application" width="100%">
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
  <img src="../docs/assets/application.gif" alt="Application demo" width="100%">
</p>
<!-- markdownlint-enable MD033 -->

## Start Here

| Task                        | Documentation                                                             |
|-----------------------------|---------------------------------------------------------------------------|
| Install the application     | [Installation](./docs/01-installation.md)                                 |
| Update an existing setup    | [Update Existing Installation](./docs/02-update-existing-installation.md) |
| Complete the first workflow | [Getting Started](./docs/03-getting-started.md)                           |

## Application Guides

| Guide                                                                     | Description                                                                       |
|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [Installation](./docs/01-installation.md)                                 | Install with Docker or run backend and UI natively.                               |
| [Update Existing Installation](./docs/02-update-existing-installation.md) | Refresh Docker images, dependencies, and services after pulling changes.          |
| [Getting Started](./docs/03-getting-started.md)                           | Create a project, set up hardware, record data, train a model, and run inference. |
| [Environment Setup](./docs/04-environment-setup.md)                       | Configure robots, cameras, and environments.                                      |
| [Recording Datasets](./docs/05-recording-datasets.md)                     | Record, review, import, and export demonstration datasets.                        |
| [Training Policies](./docs/06-training-policies.md)                       | Train model policies from recorded datasets.                                      |
| [Deploying Model Policies](./docs/07-deploying-model-policies.md)         | Run trained policies in Studio or deploy them externally.                         |

## Components

| Component                 | Description                                                    | Documentation                         |
|---------------------------|----------------------------------------------------------------|---------------------------------------|
| **[Backend](./backend/)** | FastAPI server for data management and training orchestration. | [Backend README](./backend/README.md) |
| **[UI](./ui/)**           | React web application.                                         | [UI README](./ui/README.md)           |
| **[Docker](./docker/)**   | Containerized application runtime.                             | [Docker README](./docker/README.md)   |

## See Also

- [Main Repository](../README.md) - Project overview and library quick start.
- [Library](../library/README.md) - Python SDK for programmatic usage.
