# Physical AI Studio release process

> [!NOTE]
> This document will be updated once technical implementation will be available.

## Table of Contents

- [Release artifacts](#release-artifacts)
- [Release types](#release-types)
- [Release cadence](#release-cadence)
- [Versioning](#versioning)
- [Release branches, tags and `VERSION` files format](#release-branches-tags-and-version-files-format)
  - [Release branches naming convention](#release-branches-naming-convention)
  - [GitHub tags for release candidates](#github-tags-for-release-candidates)
  - [GitHub tags for regular releases](#github-tags-for-regular-releases)
  - [GitHub tags for patch releases](#github-tags-for-patch-releases)
  - [Tagging binary release candidates](#tagging-binary-release-candidates)
  - [Tagging binary regular releases](#tagging-binary-regular-releases)
  - [Tagging binary patch releases](#tagging-binary-patch-releases)
  - [`VERSION` files format](#version-files-format)
- [Creating an application release](#creating-an-application-release)
  - [Creating an `application` regular release](#creating-an-application-regular-release)
  - [Creating an `application` patch release](#creating-an-application-patch-release)
- [Creating a `library` release](#creating-a-library-release)
  - [Creating a `library` regular release](#creating-a-library-regular-release)
  - [Creating a `library` patch release](#creating-a-library-patch-release)

## Release artifacts

The `physical-ai-studio` repository releases the following binary artifacts:

- `library` - Python package (`physicalai-train`) published to PyPI;
- `application` container images for different architectures - `physical-ai-studio-xpu`, `physical-ai-studio-cuda` and `physical-ai-studio-cpu` - pushed to `ghcr.io`.

The following container images names are used:

- `ghcr.io/open-edge-platform/physical-ai-studio-xpu`;
- `ghcr.io/open-edge-platform/physical-ai-studio-cpu`;
- `ghcr.io/open-edge-platform/physical-ai-studio-cuda`.

## Release types

Both release artifacts `library` and `application` support the following release types:

- **regular release** - created on a scheduled cycle by branching from the `main` branch;
- **patch release** - created from an existing release branch to provide security and bug fixes, based on community demand and bugfix severity.

## Release cadence

There are **different** release cadences for the `application` and the `library`, so in some cases they may be released together,
but in most cases there will be separate releases.

## Versioning

This repository uses Semantic Versioning format `<major>.<minor>.<patch>`.
As there are different release cadences for the `application` and the `library`, a single repo is used to track two different versions simultaneously.

In the following sections, we use:

- `<lib-major>.<lib-minor>.<lib-patch>` for `library` version;
- `<app-major>.<app-minor>.<app-patch>` for `application` version.

## Release branches, tags and `VERSION` files format

To identify artifact versions and tags this repo uses [PEP 440](https://peps.python.org/pep-0440/).
Although the scope of PEP 440 is Python packages, this repo extends usage where applicable for unification purposes.

### Release branches naming convention

The `main` branch is the branch for the next regular release and is under active development.

To distinguish release branches for `library` and `application`, this repository uses `lib` and `app` strings included in the release branch name.

| Artifact to be released | Release branch name format            | Example           |
| ----------------------- | ------------------------------------- | ----------------- |
| `library`               | `release/lib-<lib-major>.<lib-minor>` | `release/lib-0.1` |
| `application`           | `release/app-<app-major>.<app-minor>` | `release/app-0.1` |

### GitHub tags for release candidates

To distinguish release candidates tag for `library` and `application`, this repository uses `lib` and `app` strings included in the tag name.
Tags are used for both regular and patch release candidates.

| Artifact to be released | GitHub tag format                                    | Example                          |
| ----------------------- | ---------------------------------------------------- | -------------------------------- |
| `library`               | `lib/v<lib-major>.<lib-minor>.<lib-patch>rc<number>` | `lib/v0.1.0rc1`, `lib/v0.1.2rc1` |
| `application`           | `app/v<app-major>.<app-minor>.<app-patch>rc<number>` | `app/v0.1.0rc1`, `app/v0.1.2rc1` |

### GitHub tags for regular releases

For regular release `patch` == `0`.

| Artifact to be released | GitHub tag format                | Example      |
| ----------------------- | -------------------------------- | ------------ |
| `library`               | `lib/v<lib-major>.<lib-minor>.0` | `lib/v0.1.0` |
| `application`           | `app/v<app-major>.<app-minor>.0` | `app/v0.1.0` |

### GitHub tags for patch releases

For patch release `patch` > `0`.

| Artifact to be released | GitHub tag format                          | Example      |
| ----------------------- | ------------------------------------------ | ------------ |
| `library`               | `lib/v<lib-major>.<lib-minor>.<lib-patch>` | `lib/v0.1.2` |
| `application`           | `app/v<app-major>.<app-minor>.<app-patch>` | `app/v0.1.2` |

### Tagging binary release candidates

Semantic Versioning is used for released binary artifacts. Tags are used for both regular and patch release candidates.
These tags are included in the following way.

- Python package: tag is included in metadata that is used by the Python registry and in the Python wheel file name.
- Container image: tag is included as a container image tag.

| Artifact to be released         | Tag format                                      | Example                |
| ------------------------------- | ----------------------------------------------- | ---------------------- |
| `library` (Python package)      | `<lib-major>.<lib-minor>.<lib-patch>rc<number>` | `0.1.0rc1`, `0.1.2rc1` |
| `application` (container image) | `<app-major>.<app-minor>.<app-patch>rc<number>` | `0.1.0rc1`, `0.1.2rc1` |

### Tagging binary regular releases

For regular release `patch` == `0`.

| Artifact to be released         | Tag format                  | Example |
| ------------------------------- | --------------------------- | ------- |
| `library` (Python package)      | `<lib-major>.<lib-minor>.0` | `0.1.0` |
| `application` (container image) | `<app-major>.<app-minor>.0` | `0.1.0` |

### Tagging binary patch releases

For patch release `patch` > `0`.

| Artifact to be released         | Tag format                            | Example |
| ------------------------------- | ------------------------------------- | ------- |
| `library` (Python package)      | `<lib-major>.<lib-minor>.<lib-patch>` | `0.1.2` |
| `application` (container image) | `<app-major>.<app-minor>.<app-patch>` | `0.1.2` |

### `VERSION` files format

As there are different release cadences for the `application` and the `library`, this repo uses two separate `VERSION` files:

- `/application/VERSION` to track `application` version;
- `/library/VERSION` to track `library` versions.

There shall be no repo root-level `VERSION` file.

`pyproject.toml` uses `dynamic = ["version"]` instead of a hardcoded value.

The `main` branch always contains the version for the **next** planned release.

`/application/VERSION` content:

| Branch                                | `/application/VERSION` content        | Example |
| ------------------------------------- | ------------------------------------- | ------- |
| `main`                                | `<next-app-major>.<next-app-minor>.0` | `0.2.0` |
| `release/app-<app-major>.<app-minor>` | `<app-major>.<app-minor>.<app-patch>` | `0.1.1` |

`/library/VERSION` content:

| Branch                                | `/library/VERSION` content            | Example |
| ------------------------------------- | ------------------------------------- | ------- |
| `main`                                | `<next-lib-major>.<next-lib-minor>.0` | `0.2.0` |
| `release/lib-<lib-major>.<lib-minor>` | `<lib-major>.<lib-minor>.<lib-patch>` | `0.1.0` |

## Creating an application release

The `application` can be released independently without requiring a `library` release if there are no changes in the library.

RC should be publicly available to encourage community testing and feedback. Problems found during this period may be reported as known issues in the final release or fixed, depending on severity and community contributions.

### Creating an `application` regular release

1. Create a release branch from `main` following the naming convention `release/app-<app-major>.<app-minor>`. The part `<app-major>.<app-minor>` is taken from the `/application/VERSION` file. Release branch name example: `release/app-0.1`.
2. Increment the version in the `/application/VERSION` file in the `main` branch to prepare for the next development cycle. Example: `0.1.0` → `0.2.0`
3. Once the team decides that the release branch is ready for release candidate creation, create and push the first `application` RC GitHub tag following the naming convention `app/v<app-major>.<app-minor>.0rc1` (e.g., `app/v0.1.0rc1`) on the release branch.
4. Pushing the RC tag triggers the CI, which builds container images release candidates with RC tag (example: `0.1.0rc1`) and uploads (with signature) the release candidates into `ghcr.io`.
5. Test the RC artifacts with necessary manual tests (end-to-end / acceptance, security).
6. If changes are required, apply them to the release branch and create a new RC tag (e.g., `app/v0.1.0rc2`).
7. Once the quality of release candidate `rc-N` is acceptable, members of the team start the release process by manually launching the dedicated release workflow with the proper inputs (RC tag to be released) to promote approved release candidate to release. The release promotion workflow performs the following actions:
   - tag the approved-for-release RC commit with the release GitHub tag (e.g., `app/v0.1.0`);
   - re-tag approved release candidate with release version tag (e.g., `0.1.0rc2` -> `0.1.0`);
   - sign it with Cosign (keyless mode).

8. Compile release notes for the new version (this is currently a manual task and will be automated in future iterations).
9. Create a GitHub release and publish the release notes on the GitHub Releases page. The name of the GitHub release should allow consumers to easily find the appropriate release notes, for example: `Physical AI Studio v0.1.0`.

### Creating an `application` patch release

1. Increment the patch version (e.g., `0.1.0` → `0.1.1`) in the `/application/VERSION` file in the existing release branch (e.g., `release/app-0.1`).
2. Once the team decides that the release branch is ready for release candidate creation, create and push the first `application` RC GitHub tag (e.g., `app/v0.1.1rc1`) on the release branch.
3. Pushing the RC tag triggers the CI, which builds container images release candidates with RC tag (example: `0.1.1rc1`) and uploads (with signature) the release candidates into `ghcr.io`.
4. Test the RC artifacts with necessary manual tests (end-to-end / acceptance, security).
5. If changes are required, apply them to the release branch and create a new RC tag (e.g., `app/v0.1.1rc2`).
6. Once the quality of release candidate `rc-N` is acceptable, members of the team start the release process by manually launching the dedicated release workflow with the proper inputs (RC tag to be released) to promote approved release candidate to release. The release promotion workflow performs the following actions:
   - tag the approved-for-release RC commit with the release GitHub tag (e.g., `app/v0.1.1`);
   - re-tag approved release candidate with release version tag (e.g., `0.1.1rc2` -> `0.1.1`);
   - sign it with Cosign (keyless mode).
7. Compile release notes for the new version (this is currently a manual task and will be automated in future iterations).
8. Create a GitHub release and publish the release notes on the GitHub Releases page. The name of the GitHub release should allow consumers to easily find the appropriate release notes, for example: `Physical AI Studio v0.1.1`.

## Creating a `library` release

### Creating a `library` regular release

1. Create a release branch from `main` following the naming convention `release/lib-<lib-major>.<lib-minor>`. The part `<lib-major>.<lib-minor>` is taken from the `/library/VERSION` file. Release branch name example: `release/lib-0.1`.
2. Increment the version in the `/library/VERSION` file in the `main` branch to prepare for the next development cycle. Example: `0.1.0` → `0.2.0`
3. Once the team decides that the release branch is ready for release candidate creation, create and push the first `library` RC GitHub tag (e.g., `lib/v0.1.0rc1`) on the release branch.
4. Pushing the RC tag triggers the CI, which builds the `library` release candidate and uploads it to `test.pypi.org` with a proper version (e.g., `0.1.0rc1`).
5. Test the RC artifacts with necessary manual tests (end-to-end / acceptance, security).
6. If changes are required, apply them to the release branch and create a new RC tag (e.g., `lib/v0.1.0rc2`).
7. Once the quality of release candidate `rc-N` is acceptable, members of the team start the release process by manually launching the dedicated release workflow with the proper inputs (RC tag to be released). The release workflow performs the following actions:
   - tag the approved-for-release RC commit with the release GitHub tag (e.g., `lib/v0.1.0`);
   - rebuild the package from the tagged commit with the release version (e.g., `0.1.0`);
   - publish to PyPI (`pypi.org`) and sign it using keyless signing.

8. Compile release notes for the new version (this is currently a manual task and will be automated in future iterations).
9. Create a GitHub release and publish the release notes on the GitHub Releases page. The name of the GitHub release should allow consumers to easily find the appropriate release notes, for example: `Physical AI Library v0.1.0`.

### Creating a `library` patch release

1. Increment the patch version (e.g., `0.1.0` → `0.1.1`) in the `/library/VERSION` file in the existing release branch (e.g., `release/lib-0.1`).
2. Once the team decides that the release branch is ready for release candidate creation, create and push the first `library` RC GitHub tag (e.g., `lib/v0.1.1rc1`) on the release branch.
3. Pushing the RC tag triggers the CI, which builds the `library` release candidate and uploads it to `test.pypi.org` with a proper version (e.g., `0.1.1rc1`).
4. Test the RC artifacts with necessary manual tests (end-to-end / acceptance, security).
5. If changes are required, apply them to the release branch and create a new RC tag (e.g., `lib/v0.1.1rc2`).
6. Once the quality of release candidate `rc-N` is acceptable, members of the team start the release process by manually launching the dedicated release workflow with the proper inputs (RC tag to be released). The release workflow performs the following actions:
   - tag the approved-for-release RC commit with the release GitHub tag (e.g., `lib/v0.1.1`);
   - rebuild the package from the tagged commit with the release version (e.g., `0.1.1`);
   - publish to PyPI (`pypi.org`) and sign it using keyless signing.

7. Compile release notes for the new version (this is currently a manual task and will be automated in future iterations).
8. Create a GitHub release and publish the release notes on the GitHub Releases page. The name of the GitHub release should allow consumers to easily find the appropriate release notes, for example: `Physical AI Library v0.1.1`.
