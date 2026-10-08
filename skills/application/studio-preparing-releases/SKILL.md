---
name: studio-preparing-releases
description: Prepares and publishes versioned releases of `physicalai-studio` and `physicalai-train`. Triggered by requests to bump either package, prepare a release PR or notes, cut `app/vX.Y.Z` or `lib/vX.Y.Z` tags, or verify GitHub Releases and PyPI publications.
---

# Preparing Studio releases

Treat `physicalai-studio` and `physicalai-train` as separate distributions with separate versions. Their source-of-truth versions are `application/backend/pyproject.toml` and `library/pyproject.toml`, respectively. Studio runtime code reads its installed distribution metadata; workflows read the relevant project metadata. Do not introduce a shared version file or assume both packages must always have the same version.

## Workflow

1. **Check release state.** Inspect the worktree, current branch, `upstream/main`, the previous `app/v*` and `lib/v*` tags, and whether the requested version already exists on GitHub or PyPI. Prepare from a branch based on current `upstream/main`; preserve unrelated work. Done when the target version is unused and the exact release scope is clear.
2. **Bump package metadata.** Set `project.version` in the pyproject for each package being released. For a coordinated release, set the `physicalai` and `physicalai-train` minimum requirements only where the released code requires those APIs. Do not bump curated plugin requirements unless the plugin package has a compatible published release. Done when project metadata and dependency floors express the intended compatibility.
3. **Regenerate and validate.** Run `uv lock` in `library/` and `application/backend/` as applicable, then `uv lock --check` in both. Build the library with `cd library && uv build --no-sources`; validate the Studio wheel with `bash application/backend/scripts/build_package.sh` when UI dependencies are available (this build may require network access). Run focused tests for changed runtime settings and relevant package checks. Done when locks are current, builds succeed, and the installed Studio version comes from `physicalai-studio` distribution metadata.
4. **Prepare notes and PR.** Read the previous GitHub release notes and summarize user-facing changes since the previous tag separately for the library and app. Include compatibility/upgrade notes and install instructions. Open a signed release-prep PR against `main`, wait for required checks and code-owner reviews, and merge normally. Done when the release-prep commit is on `upstream/main` and its checks are green.
5. **Publish only when explicitly requested.** Confirm the merged main commit, confirm the tags do not exist, and create the library release first:

   ```bash
   gh release create "lib/v${VERSION}" --repo open-edge-platform/physical-ai-studio \
     --target "$MAIN_SHA" --title "physicalai-train v${VERSION}" --notes-file "$LIBRARY_NOTES"
   ```

   Wait for the `Publish PyPI` workflow to succeed and verify `physicalai-train==${VERSION}` on PyPI. Only then create the app release with `gh release create "app/v${APP_VERSION}" --repo open-edge-platform/physical-ai-studio --target "$APP_SHA" --title "Physical AI Studio v${APP_VERSION}" --notes-file "$APP_NOTES"`; wait for its publish workflow and verify `physicalai-studio==${APP_VERSION}`. Tags must point at merged main commits. Never bypass required review or CI to publish. Done when both GitHub releases and both PyPI versions resolve to the expected commit/version.
