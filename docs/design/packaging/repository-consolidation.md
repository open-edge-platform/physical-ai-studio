# Repository Consolidation

## Summary

This document proposes that the Physical AI runtime, the training library, the Studio application and the
first-party plugins are developed in one repository, `open-edge-platform/physical-ai-studio`. Published packages,
their names and their installation commands do not change; only the place where they are developed does.

The proposal has four consequences:

- A change that spans the runtime, the training library and Studio is made, reviewed and tested in one pull request.
- The repository is organized by role, with one directory per package.
- All Python packages use the same version of each dependency, recorded in one lock file.
- All packages are released from one release pull request, extending the process proposed in
  [Robot Integration Packaging][robot-doc].

The future of `openvinotoolkit/physicalai` is a separate decision that the proposed structure does not depend on. The
repository can be archived or kept as an automatically updated copy; see
[Relationship to openvinotoolkit/physicalai](#relationship-to-openvinotoolkitphysicalai).

## Background

The Physical AI workflow is delivered by four groups of packages in two repositories:

| Component                                          | Packages                         | Repository                              | Source size                            |
| -------------------------------------------------- | -------------------------------- | --------------------------------------- | -------------------------------------- |
| Runtime: inference, cameras, robots, control loop  | `physicalai`                     | `openvinotoolkit/physicalai`            | 27k lines of Python                    |
| Robot plugins and the Plugin SDK                   | 6 `physicalai-*-plugin` packages | `openvinotoolkit/physicalai`            | 14k lines of Python                    |
| Training library: data, policies, training, export | `physicalai-train`               | `open-edge-platform/physical-ai-studio` | 57k lines of Python                    |
| Studio application: backend and user interface     | `physicalai-studio`              | `open-edge-platform/physical-ai-studio` | 31k lines of Python, 34k of TypeScript |

The training library and Studio depend on the runtime. The runtime originated in the Studio repository and was moved
to `openvinotoolkit/physicalai` in May 2026 ([runtime #107][rt107], [Studio #603][st603]); since then, Studio has
depended on a specific commit of the runtime repository. The [packaging strategy][two-repo] written for that move
anticipated a later consolidation, subject to approval.

## Motivation

**The repositories are maintained by one team.** From May to early October 2026, 135 of the 141 commits to the
runtime (96%) were made by people who also contribute to Studio.

**Changes frequently span both repositories.** Since the move, Studio has updated its runtime reference 22 times,
usually to a runtime commit made within the previous three days. Sixteen of these updates also changed Studio code;
in each case, one change was split into two pull requests, two reviews and two merges in a fixed order.

**Incompatibilities are found after merging.** Runtime pull requests are not tested against Studio. An incompatible
change therefore surfaces only when Studio next updates its reference, as in [Studio #644][st644], which fixed an
inference failure caused by a change to a runtime return type. Automated reference updates were tried
([#813][st813]) and disabled six days later ([#838][st838]).

**Published requirements differ from what is tested.** `physicalai-train` 0.2.0 requires `physicalai>=0.1.0`, but
nine of its modules import `physicalai.config`, which first appeared in `physicalai` 0.2.0. `physicalai-studio` 0.2.0
declares no minimum runtime version.

**Shared dependencies are kept consistent by hand.** Transport libraries that must match between processes are pinned
to exact versions, with comments asking maintainers to keep them in sync with Studio. Studio delays the adoption of
new dependency releases by seven days and needs exceptions for packages the runtime has already adopted. The
repositories maintain three separate lock files.

Two further issues do not depend on the split but are easier to resolve during it:

- **Overlapping import paths.** `physicalai` and `physicalai-train` both add modules to `physicalai.cli`,
  `physicalai.benchmark` and `physicalai.inference.adapters`; the last relies on a special import mechanism in the
  runtime.
- **Unscoped application modules.** The published `physicalai-studio` package installs 28 top-level modules, such as
  `api`, `core` and `utils`, which can conflict with other installed packages.

## Goals and Non-Goals

Goals:

1. Develop, review and test changes across packages in one pull request.
2. Keep `physicalai` installable on a robot without PyTorch or other training dependencies.
3. Give every directory, package and import path a single owner.
4. Use one version of each dependency across all packages.
5. Publish only version requirements that have been tested.

Non-goals:

- Renaming published packages, the `physicalai` command or installation commands.
- Merging the runtime and the training library into one package, or splitting the runtime into smaller packages.
- Deciding the future of `openvinotoolkit/physicalai`.
- Changing the architecture of the Studio application.

## Proposed Structure

```text
physical-ai-studio/
├── packages/
│   ├── runtime/              physicalai, including the Plugin SDK
│   └── train/                physicalai-train
├── plugins/
│   └── robots/
│       ├── so101/            planned in Robot Integration Packaging
│       ├── bimanual-so101/
│       ├── trossen/          planned in Robot Integration Packaging
│       ├── rebot-b601/
│       ├── stararm/
│       ├── mujoco-so101/
│       └── lerobot/
├── apps/
│   └── studio/
│       ├── backend/          physicalai-studio
│       ├── ui/
│       └── deploy/           container and cloud deployment files
├── tests/                    tests that involve more than one package
├── docs/                     one documentation site
├── examples/                 examples and notebooks
├── skills/                   agent skills
├── tools/                    development, release and synchronization scripts
├── pyproject.toml            workspace definition and shared settings
└── uv.lock                   one lock file
```

Directories are named by role; package names are unchanged. Each package directory contains `pyproject.toml`,
`README.md`, `LICENSE`, `CHANGELOG.md`, `src/` and `tests/`.

| Directory               | Package                    | Import path                                                                   | Depends on                                                    |
| ----------------------- | -------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `packages/runtime`      | `physicalai`               | `physicalai.{inference, capture, robot, runtime, config, cli, benchmark}`     | —                                                             |
| `packages/train`        | `physicalai-train`         | `physicalai.{train, policies, data, eval, gyms, export, transforms, devices}` | `physicalai`                                                  |
| `plugins/robots/<name>` | `physicalai-<name>-plugin` | `physicalai_<name>_plugin`                                                    | `physicalai`                                                  |
| `apps/studio/backend`   | `physicalai-studio`        | `physicalai_studio`                                                           | `physicalai`, `physicalai-train`; plugins installed on demand |

As proposed in Robot Integration Packaging, the Plugin SDK becomes part of `physicalai`; `physicalai-studio-plugin`
remains as a compatibility package for one release.

## Design Rules

1. **Dependencies point in one direction.** `physicalai` depends on no other package in the repository; the plugins
   and `physicalai-train` depend on `physicalai`; Studio depends on all of them. Automated checks reject imports in
   the opposite direction and confirm that installing `physicalai` alone does not install PyTorch.
2. **Each import path belongs to one package.** Only the top-level `physicalai` name is shared. Modules that
   `physicalai-train` adds to runtime paths move to training paths ([Required Changes](#required-changes)). The
   command-line subcommands and inference backends among them are registered through entry points (the standard
   Python mechanism by which one package declares functionality that another discovers at run time), so the
   `physicalai fit` command and the PyTorch and ExecuTorch backends are unaffected.
3. **Applications have their own namespace.** Studio's backend code moves into a `physicalai_studio` package, which
   installs one top-level name instead of 28.
4. **One workspace, one lock file.** All Python packages belong to one uv workspace (packages developed and installed
   together from source) with one lock file (the exact version of every dependency). The existing lock files suggest
   this is feasible: Studio's backend already resolves the runtime, the training library and Studio together (279
   packages), and the runtime resolves with the Plugin SDK and four plugins (155 packages). The PyTorch variants need
   common names (`cu128` in `physicalai-train`, `cuda` in Studio). Because the lock file covers only the Python
   versions that all packages support (3.12 and 3.13), runtime and plugin support for 3.11 and 3.14 is verified by
   installing the built packages. A package that cannot be resolved with the others, such as the RoboCasa environment
   today, keeps its own lock file outside the workspace.
5. **Tests are placed by scope.** Each package keeps its own tests. The root `tests/` directory covers the contracts
   between packages: exporting each policy with `physicalai-train` and loading it with `physicalai`, plugin
   registration in Studio, and the `physicalai` command with the subcommands contributed by `physicalai-train`.
6. **Documentation and agent skills are unified.** The four documentation trees become one site organized by task:
   deploying a policy, training, using Studio, adding a robot, and reference. Package READMEs become short PyPI
   descriptions that link to it.

## Plugins

Plugins are grouped by the kind of extension they provide. `plugins/robots/` holds all robot integrations;
`plugins/cameras/` and `plugins/policies/` are added with the first plugin of each kind. LeRobot uses the same
division (robots, cameras, policies, teleoperators and environments), including for third-party plugins.

- **Placement.** A plugin sits under the kind it mainly provides and may include closely related functionality used
  in the same place; the MuJoCo simulation, for example, provides a robot and its cameras.
- **Separation.** Robot-side and training-side functionality are not combined in one plugin. Robot and camera
  plugins must install on a robot without PyTorch, while policy plugins require `physicalai-train`.
- **Extension points.** Each kind has one extension point, defined by the package that owns its interface. Robots
  have one: they are loaded by class path and listed in Studio through the `physicalai.studio.catalog_plugins`
  entry-point group. Cameras and policies do not yet: camera backends are selected from a fixed list in
  `create_camera()`, and there is no way to list policies provided by other packages. Adding them is a prerequisite
  for camera and policy plugins.
- **Conformance tests.** Each kind provides a test suite in its core package that every plugin runs, starting from
  the existing `verify_robot()` for robots.

## Versioning and Releases

Each package keeps its own version, and all packages are released from one release pull request, as proposed in
Robot Integration Packaging. The process is extended to `physicalai-train` and `physicalai-studio` and publishes in
dependency order: `physicalai`, the plugins, `physicalai-train`, then `physicalai-studio` with its container images.

Within the workspace, packages use each other's source code regardless of declared version requirements. The release
checks therefore build each package at its new version, install it from the built files and PyPI only, and test it
against the oldest and newest versions it allows. These checks would have rejected the `physicalai-train`
requirement described in [Motivation](#motivation).

Compatibility between a deployed runtime and a newer training library is governed by the export format rather than by
package versions. The manifest written by `physicalai-train` already carries its own version (currently 1.0);
`physicalai` should state which manifest versions it can load.

## Automated Checks

Checks run for the packages affected by a change and for the packages that depend on them:

| Change in                               | Checks run for                                                     |
| --------------------------------------- | ------------------------------------------------------------------ |
| `packages/runtime`                      | Runtime, plugins, training library, Studio and cross-package tests |
| `packages/train`                        | Training library, Studio and cross-package tests                   |
| `plugins/<kind>/<name>`                 | That plugin and Studio's plugin tests                              |
| `apps/studio/backend`, `apps/studio/ui` | The changed part of Studio                                         |
| `pyproject.toml`, `uv.lock`             | Everything                                                         |

## Required Changes

| Current                                                                             | Proposed                                                  | Effect on users                                           |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| `physicalai.cli` modules of `physicalai-train` (subcommands and helpers)            | `physicalai.train.cli`                                    | None; registered through entry points                     |
| `physicalai.inference.adapters` modules of `physicalai-train` (PyTorch, ExecuTorch) | A training path, for example `physicalai.export.adapters` | None; registered through entry points                     |
| `physicalai.benchmark.gyms` of `physicalai-train`                                   | A training path, for example `physicalai.eval.benchmark`  | Old path kept for one release; configuration files use it |
| Private `physicalai.cli._spec`, used by `physicalai-train`                          | Public module in `physicalai`                             | None                                                      |
| Top-level modules of the Studio backend (`api`, `core`, …)                          | `physicalai_studio`                                       | None for users of the application                         |
| Studio's reference to a runtime commit                                              | Workspace dependency                                      | None                                                      |

## Migration

Each step leaves the repository releasable.

1. Agree on the structure, directory owners and supported Python versions, and pause merges to the runtime repository
   during the import.
2. Import the runtime repository with its full history mapped to the new directories (for example with
   `git filter-repo`), so that every file keeps its history. The import includes 178 files stored with Git Large File
   Storage.
3. Move `library/` to `packages/train/` and `application/` to `apps/studio/`; Git follows the renames.
4. Create the workspace and lock file, add a license file to every package that lacks one, replace the runtime commit
   reference, and align the PyTorch variant names.
5. Apply the [Required Changes](#required-changes), with compatibility aliases where noted.
6. Merge documentation, agent skills and examples, and configure the automated checks.
7. Extend the release process to all packages, configure PyPI publishing according to the decision below, and release
   every package through the new process, first to TestPyPI.

The work in Robot Integration Packaging can precede or follow these steps; completing its single release process
first reduces the changes made during consolidation.

## Relationship to openvinotoolkit/physicalai

Either of two options is compatible with the proposed structure.

**Archive.** The repository is archived with a notice pointing to the consolidated repository, which takes over PyPI
publishing. Issues cannot be moved between organizations directly; they can be moved after the repository is
transferred to `open-edge-platform`, or closed with a reference. Package names and release history on PyPI are
unaffected.

**Automatically updated copy.** The repository remains the public home of `physicalai`, with its README, releases,
issue tracker and PyPI links under `openvinotoolkit`, but its content is maintained by automation:

- **Content.** The copy contains a fixed list of directories from the consolidated repository at the same paths:
  `packages/runtime`, the Plugin SDK, `plugins/robots`, and the runtime's documentation, examples and agent skills.
  Files that exist only in the copy, such as its README, contribution guide, workflows and workspace definition, are
  kept in `tools/mirror/openvino/` in the consolidated repository.
- **Updates.** After each merge, an export job appends one commit to the copy for each change to these directories,
  keeping the original author and message and recording the source commit. Because the copy's history is only
  extended, its existing commits, tags and forks remain valid. Release tags of the runtime and plugins are forwarded,
  and the copy publishes them to PyPI with its current configuration.
- **Contributions.** A pull request opened on the copy is applied as a patch to a branch of the consolidated
  repository. There it is reviewed and merged without squashing, so that the contributor, rather than the bot that
  opened the pull request, is recorded as the author; the change then returns to the copy. Issues stay on the copy,
  and a reference such as `Fixes openvinotoolkit/physicalai#42` closes the issue when the change reaches it.

A prototype of both directions required about 200 lines of code (see the
[appendix](#appendix-measurements-and-prototype)). [Copybara][copybara] implements the same model with more complete
handling of renamed files and conflicts. Operating the copy requires a GitHub App, approved by both organizations,
with sole write access to the copy's main branch, as well as the transfer of Git LFS files. The copy's layout changes
once, so links to files at their former paths no longer resolve. Its commits are not associated with pull requests in
the copy, which affects the code-review check of OpenSSF Scorecard.

## Alternatives Considered

**Keep two repositories and automate coordination.** Runtime pull requests would run Studio's tests, and Studio's
runtime reference would be updated automatically. This reduces manual work, but changes across the boundary still
need two reviews and two merges, and published requirements remain untested.

**Keep the runtime in one top-level directory.** A `runtime/` directory beside `library/` and `application/` was
considered to simplify exporting the runtime to `openvinotoolkit/physicalai`. The prototype showed that the proposed
layout can be exported as simply, so this alternative offers no advantage, and it would organize the repository by
history rather than by role.

**Consolidate in `openvinotoolkit/physicalai`.** This is technically equivalent; the choice is organizational. This
document assumes the Studio repository, which holds most of the code and contributors.

**Separate lock files for the runtime and Studio.** This preserves the runtime's full Python range in its lock file,
at the cost of keeping shared versions consistent by hand.

**One package with optional dependencies.** This removes the boundary between runtime and training. A separate
package remains the most reliable way to keep the robot-side installation free of PyTorch.

**One version for all packages.** Not adopted in Robot Integration Packaging, because every fix would release every
package.

## Trade-offs

- The repository is larger, including for contributors who work only on the runtime, and automated checks must be
  limited to affected packages to keep their duration reasonable.
- With one lock file, a dependency upgrade affects all packages at once.
- Development environments cover only the Python versions shared by all packages.
- Imported runtime commits receive new identifiers; links to existing commits refer to the original repository.
- Moving import paths requires a deprecation period.

## Open Questions

1. Should `openvinotoolkit/physicalai` be archived or kept as an automatically updated copy?
2. If the copy is kept, should the runtime and plugins be published to PyPI from the copy, which preserves the current
   configuration, or from the consolidated repository?
3. Should the repository be renamed to reflect its broader scope, for example to `physicalai`? GitHub redirects the
   previous name.
4. Which release tag scheme should all packages use? The runtime uses `v*` and `<package>-v*`; Studio uses `lib/v*`
   and `app/v*`.
5. Who reviews `packages/runtime`? Code-owner rules can reference only teams in the repository's own organization.
6. Should the consolidation precede or follow Robot Integration Packaging?

## Out of Scope

- A naming scheme for plugin packages.
- Extension points for camera and policy plugins.
- Guidelines and tooling for third-party plugin authors.

## References

- [Robot Integration Packaging][robot-doc]
- [Physical-AI Packaging Strategy][two-repo], the two-repository design
- [Modular Packages in One Repo][modular]
- [uv workspaces][uv-ws], [Copybara][copybara], [git filter-repo][filter-repo]
- Runtime [#107][rt107]; Studio [#603][st603], [#644][st644], [#813][st813], [#838][st838]

## Appendix: Measurements and Prototype

Measured on 2026-10-07 from both repositories and PyPI:

| Measurement                                                 | Value                                      |
| ----------------------------------------------------------- | ------------------------------------------ |
| Runtime commits since 2026-05-01, excluding automated ones  | 141, of which 135 by Studio contributors   |
| Studio updates of the runtime reference since the move      | 22                                         |
| Updates that also changed Studio code                       | 16 (11 in `library/`, 8 in `application/`) |
| Updates referencing a runtime commit at most three days old | 18 of 19 measured                          |
| Runtime version required by `physicalai-train` 0.2.0        | Declares `>=0.1.0`; requires 0.2.0         |
| Top-level modules installed by `physicalai-studio` 0.2.0    | 28                                         |
| Lock files (packages resolved)                              | 3 (155, 264 and 279)                       |
| Documentation trees                                         | 4                                          |

A prototype built the proposed structure from local copies of both repositories and maintained a copy of
`openvinotoolkit/physicalai`, starting from its full existing history. It showed that:

- uv does not support nested workspaces, so the runtime cannot remain a separate workspace inside a repository-wide
  one.
- Within a workspace, packages ignore each other's declared version requirements; a requirement of `>=0.2.0` was
  satisfied by version 0.1.0.
- The runtime package could not be built until it had its own license file, because its packaging configuration
  refers to one in its own directory.
- An exporter of 145 lines kept the copy up to date. Its first run appended one commit to the existing history without
  rewriting it. Later runs exported only changes to the copy's directories, kept authors, co-authors and issue
  references, forwarded a release tag, and found nothing to export when repeated.
- The copy was usable on its own: its lock file resolved the same 155 packages as the current runtime repository, and
  `physicalai` built with a version derived from the copy's tags.
- An importer of 56 lines applied a pull request opened on the copy to the consolidated repository, including a change
  to a file that exists only in the copy, after the same file had changed in the consolidated repository. Once merged,
  the change returned to the copy under the contributor's name.

[robot-doc]: https://github.com/openvinotoolkit/physicalai/blob/docs/design/docs/design/packaging/robot-integrations.md
[two-repo]: https://github.com/openvinotoolkit/physicalai/blob/docs/design/docs/design/packaging/physical-ai-two-repo-options.md
[modular]: https://github.com/openvinotoolkit/physicalai/blob/docs/design/docs/design/packaging/modular-packages-in-one-repo.md
[uv-ws]: https://docs.astral.sh/uv/concepts/projects/workspaces/
[copybara]: https://github.com/google/copybara
[filter-repo]: https://github.com/newren/git-filter-repo
[rt107]: https://github.com/openvinotoolkit/physicalai/pull/107
[st603]: https://github.com/open-edge-platform/physical-ai-studio/pull/603
[st644]: https://github.com/open-edge-platform/physical-ai-studio/pull/644
[st813]: https://github.com/open-edge-platform/physical-ai-studio/pull/813
[st838]: https://github.com/open-edge-platform/physical-ai-studio/pull/838
