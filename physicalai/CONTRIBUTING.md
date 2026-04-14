# Contributing to Physical AI Runtime

## Getting Started

This repo contains the Physical AI runtime library.

## Code Quality

We use [prek](https://prek.j178.dev/) (Rust-based pre-commit) for code quality hooks.

### Install prek

```bash
# Using cargo
cargo install prek

# Or using the install script
curl -fsSL https://prek.j178.dev/install.sh | sh
```

### Install Git Hooks

```bash
prek install
```

### Run Hooks Manually

```bash
# All files
prek run --all-files
```

## Commit Messages

Use [conventional commits](https://www.conventionalcommits.org/):

- `feat:` - new features
- `fix:` - bug fixes
- `docs:` - documentation changes
- `refactor:` - code refactoring
- `test:` - adding tests
- `chore:` - maintenance tasks

Write clear, concise messages. Reference issue numbers when applicable.

## Pull Requests

- Follow conventional commit format for PR title
- Fill out the PR template completely
- Provide usage examples for new features
- Note any breaking changes

## Coding Standards

See [.github/copilot-instructions.md](./.github/copilot-instructions.md) for detailed coding standards, style guides, and best practices.

---

## License

Physical AI Runtime is licensed under the terms in [LICENSE](./LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Sign Your Work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify the below (from [developercertificate.org](http://developercertificate.org/)):

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

```text
Signed-off-by: Joe Smith <joe.smith@email.com>
```

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your commit automatically with `git commit -s`.
