---
name: env-setup
description: >-
  Explain, inspect, configure, and troubleshoot Cloud Agent development
  environments. Use when the user asks about environment setup,
  changing/improving the environment, or triggering/testing a build.
environments:
  - cloud
---
# Cloud Agent Environment Setup

Use this skill to teach users how Cloud Agent environments work and, when requested, to inspect or improve the current repository's setup. Match the response to the request: explain a concept, audit configuration, make a focused patch, or troubleshoot a failed setup.

## Resources

| Workflow | Reference |
| --- | --- |
| Create or fully set up an environment (snapshot, build-test when available, propose) | [Set up a new environment](references/create-environment.md) |
| Update a repository-managed environment | [Update a repository-managed environment](references/update-repo-managed-environment.md) |
| Update a DB-managed environment | [Update a DB-managed environment](references/update-db-managed-environment.md) |
| Verify an existing environment's builds | [Verify an environment's builds](references/migrate-to-builds.md) |

## Mental Model

A Cloud Agent runs in an isolated VM with two layers:

1. **Base environment**: a saved snapshot, Dockerfile-built image, explicit image, or Cursor's default image supplies the operating system, system packages, and toolchains.
2. **Repository bootstrap**: `install` prepares dependencies and source-derived files after checkout; `start` brings up runtime services on every agent boot.

Most environments run on **environment builds**. Cursor periodically (and on demand) boots a build pod, checks out each repository's default branch, runs `install` to completion, and snapshots the disk. New agents boot from the latest successful build, skip clone and `install`, and check out their own branch on top. Without a usable build (none has succeeded yet), agents boot Cursor's default image instead of the configured base environment and skip `install`.

Configuration changes affect the next build and newly started agents, never an already-running agent.

## How install, start, and terminals run

| Field | During a build | When an agent boots | Blocks the agent? | On failure |
| --- | --- | --- | --- | --- |
| `install` | Runs once on the build pod, before the snapshot; 90-minute budget. | From a build: never runs. Without a usable build: skipped, and the agent boots Cursor's default image instead. | No. Only the build's snapshot waits for it. | Nonzero exit fails the build; agents keep booting from the previous successful build. |
| `start` | Does not run. | Runs on every boot. Exception: a warm-forked agent skips it when this run's environment cannot be delivered to the relaunch. Written to a script and launched detached with `bash -l` from `/workspace`, never awaited, no timeout. | No. The agent may begin before services are ready. | Nothing fails. Output goes to `/tmp/cursor/start-user/start-user.log`; the exit code is written to `start-user.status` only when the script exits. A missing `/tmp/cursor/start-user/` means `start` did not run on this boot. |
| `terminals` | Not part of the snapshot. | Launched concurrently with `start` on every boot that runs `start`, so they race it rather than waiting for it. Each entry becomes a detached tmux session (`tmux new-session -d` plus `send-keys "<command>"`) named after the entry, lowercased, with non-alphanumerics replaced by `_` (`dev-server` shows as `dev_server` in `tmux ls`). **Ignored entirely when `start` is absent.** | No. | Nothing is reported; output lives only in the tmux pane. |

Consequences:

- Processes never survive a snapshot. Anything that must be running belongs in `start`, not `install`.
- `install` output reflects the build's default-branch commit. An agent on a branch with different dependencies must refresh them itself; do not move dependency installs into `start` to compensate.
- A successful build proves only the image and `install`. `start` and `terminals` first run when an agent boots, so verify them in a fresh agent booted from that build.
- `trigger-environment-build` and `propose-environment-json` accept only `install`, `start`, and (for builds) `snapshot`. A DB-managed environment cannot get `terminals` through this skill.
- `terminals` commands are pasted unescaped into `send-keys "<command>"` inside a generated bash script, so `"` breaks the command, and `$VAR`, `$(...)`, backticks, and backslashes are expanded by that script before the command reaches the tmux session.
- `terminals` that depend on services `start` brings up must wait for them themselves.

## Choosing install, start, or terminals

- **`install`**: work that produces files: package installs, compilation, code generation, local configuration. Idempotent, non-interactive, and it must exit.
- **`start`**: everything that must be running while the agent works: daemons (Docker, databases) and long-running dev servers, workers, and watchers. This is where the platform expects long-running servers. Because nothing waits on `start`, it may keep a server in the foreground for the life of the agent.
- **`terminals`**: only when a repository-managed `environment.json` already uses them or the user asks for them, and always together with `start`. To give each server a named tmux session, create the sessions from the `start` script instead; you control quoting, logs, and readiness there.
- **Agent-session processes**: servers the agent launches ad hoc while working, rather than on every boot, run in tmux sessions per the shell tool guidance and stay out of `environment.json`.

Write `start` to be observable and safe to rerun:

- No dependency installs or source generation; they slow every boot and belong in `install`.
- Skip services that are already up, and clean stale PID or socket files.
- Wait for readiness of what the agent needs (for example `pg_isready`) so a failure shows in `start-user.log`.
- Either keep the final server in the foreground with `exec` (then a `start-user.status` file means the server died) or start servers in tmux and return (then `0` means startup succeeded).
- Multi-line inline scripts work; for repository-managed configs prefer a committed script.

### Runtime user and PATH

- `ubuntu:24.04` (and Cursor's default image) already has an `ubuntu` user, so `useradd ... ubuntu` fails the image build. On that base, do not create it; set `"user": "ubuntu"` in `environment.json` and use the existing account. Whatever `user` names must exist in the image.
- `install` and `start` run as `bash -lc` / `bash -l`: non-interactive login shells. Ubuntu's stock `~/.bashrc` returns early when not interactive, so PATH lines that installers append to it (Bun, many version managers) never load, and `install` fails with `bun: command not found`. Each phase is a separate process, so running the installer earlier does not help later commands either.
- Put tools on the default PATH: install into or symlink into `/usr/local/bin`. Do not rely on `~/.bashrc`, `~/.profile`, or a version manager that only edits rc files. Dockerfile `ENV PATH=...` works on Ubuntu bases, but Debian-based images (including `node:*-bookworm`) reset PATH in login shells, so prefer `/usr/local/bin`.

Bun in a Dockerfile:

~~~dockerfile
USER ubuntu
RUN curl -fsSL https://bun.sh/install | bash
USER root
RUN ln -sf /home/ubuntu/.bun/bin/bun /usr/local/bin/bun
USER ubuntu
~~~

Bun from `install` on the default image (passwordless `sudo`):

~~~bash
command -v bun >/dev/null || curl -fsSL https://bun.sh/install | bash
sudo ln -sf "$HOME/.bun/bin/bun" /usr/local/bin/bun
bun install --frozen-lockfile
~~~

### Examples

One dev server, held in the foreground by `start`:

~~~json
{
  "install": "npm ci",
  "start": "exec npm run dev -- --host 0.0.0.0 --port 3000"
}
~~~

A database plus two servers in named tmux sessions, with `"start": "bash .cursor/start.sh"`:

~~~bash
#!/usr/bin/env bash
set -euo pipefail
sudo service postgresql start
until pg_isready -q; do sleep 1; done
for svc in web worker; do
  tmux has-session -t "$svc" 2>/dev/null ||
    tmux new-session -d -s "$svc" "pnpm run $svc 2>&1 | tee /tmp/$svc.log"
done
~~~

Broken: no `start`, so the dev server never launches:

~~~json
{
  "install": "npm ci",
  "terminals": [{ "name": "dev", "command": "npm run dev" }]
}
~~~

### Diagnose misplaced work

1. Identify the phase from build logs, setup logs, or `/tmp/cursor/start-user/`. Do not infer it from the script name.
2. `install` launches a server, watcher, worker, or daemon: move it to `start`.
3. `start` installs packages, compiles, or regenerates source: move that to `install`.
4. `terminals` never appear (`tmux ls` is empty): `start` is missing, or a command contains quotes, `$`, backticks, or backslashes that the generated script mangled. Move the processes into `start`.
5. The build succeeds but services are missing in agents: read `start-user.log` and `start-user.status` in a fresh agent; the build never ran `start`. If `/tmp/cursor/start-user/` is missing, `start` did not run on this boot.
6. "Address already in use" or duplicate workers: `start` is not idempotent.

## Configuration Sources and Precedence

Cursor resolves environment configuration in this order:

1. `.cursor/environment.json` from the repository revision used to start the agent.
2. A personal saved environment for the repository.
3. A team saved environment for the repository.

The first available source wins. A committed `.cursor/environment.json` therefore overrides dashboard-managed personal and team environments. Before proposing a change, determine which source the current agent actually used. Prefer the Cloud Agent environment-info tool when available; otherwise inspect the repository and explain what cannot be confirmed.

Repository-defined environments are versioned with code and are best when the setup should follow branches and pull requests. Dashboard-managed environments are useful for interactive setup, secrets, and reusable saved snapshots without committing configuration to the repository.

## Common environment.json Fields

Use the current public schema instead of relying on a memorized exhaustive field list:

- Schema: https://cursor.com/schemas/environment.schema.json
- Setup guide: https://cursor.com/docs/cloud-agent/setup
- Environment settings: https://cursor.com/docs/cloud-agent/settings

Use the schema for validation, but do not add a `$schema` property to `.cursor/environment.json`; the current schema rejects undeclared fields.

Common fields:

| Field | Purpose |
| --- | --- |
| `name` | Human-readable environment name. |
| `user` | User that runs commands in the environment. It must exist in the image. |
| `build.dockerfile` | Dockerfile path, relative to the directory containing `environment.json`. |
| `build.dockerfileContents` | Inline Dockerfile used by supported dashboard-managed flows. Prefer a repository file for repo-managed environments. |
| `build.context` | Docker build context, also relative to that directory. It defaults to `.cursor`. |
| `image` | Explicit container image reference. |
| `snapshot` | Saved base-environment snapshot ID. |
| `install` | Idempotent repository bootstrap run after checkout; baked into builds. |
| `start` | Per-boot script launched detached on every agent boot; never awaited and never fails the agent. |
| `terminals` | Named tmux sessions launched alongside `start`; ignored when `start` is absent. |
| `ports` | Container ports to expose. |
| `repositoryDependencies` | Additional repositories that must be in the generated GitHub token's access scope. |
| `agentCanUpdateSnapshot` | Whether the agent may update a snapshot-backed environment. |
| `chromeExecutablePath` | Browser executable to use for computer-use flows. |
| `egressMode` and `egressAllowlist` | Network-policy inputs. Treat dashboard network settings and inherited policy as part of the effective result. |
| `mcpServerAllowlist` | MCP server URL or command entries permitted by the environment policy. |

Choose one base-image strategy: Dockerfile build, explicit image, or snapshot. Do not combine them and rely on implementation precedence. Check the live schema before adding less-common fields because product behavior can evolve faster than this skill.

## Search Current Documentation

When setup involves a third-party service, hosted platform, or a stack you are not deeply familiar with, look up the provider's current official documentation with web search and fetch tools when available instead of relying on memorized knowledge. Install methods, CLI names, package versions, local-development workflows, and auth flows change faster than any memorized snapshot:

- Supabase: check the current `supabase` CLI install method and local-stack (`supabase start`) requirements before writing install or start commands.
- Firebase, Stripe, and similar hosted services: confirm the current CLI, emulator, or local-proxy story rather than assuming one exists.

Prefer official provider documentation over secondhand sources. If web search or network access is unavailable in the current egress mode, say so, rely on what the repository itself pins (lockfiles, tool-version files, setup scripts), and avoid inventing provider-specific commands.

## Dockerfile and Build Context

`build.dockerfile` and `build.context` are resolved relative to the directory containing `.cursor/environment.json`:

- `"dockerfile": "Dockerfile"` means `.cursor/Dockerfile`.
- Omitting `context` uses `.cursor` as the build context.
- `"context": "."`, `"./"`, or `".."` makes the repository root available as the build context.

Cursor checks out the requested repository revision separately after preparing the base environment. Avoid copying the whole repository into the image: it creates stale duplicate source, weakens layer caching, and makes branch behavior confusing. Use the Dockerfile for operating-system packages and stable tool versions; use `install` for dependency installation tied to lockfiles and source.

### Minimal default-image setup

~~~json
{
  "name": "My project",
  "install": "npm ci"
}
~~~

### Custom Dockerfile

`.cursor/environment.json`:

~~~json
{
  "name": "My project",
  "build": {
    "dockerfile": "Dockerfile",
    "context": ".."
  },
  "user": "ubuntu",
  "install": "./scripts/cloud-agent-install.sh"
}
~~~

`.cursor/Dockerfile`:

~~~dockerfile
FROM ubuntu:24.04

ARG NODE_VERSION=22.12.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git xz-utils build-essential python3 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
       | tar -xJ -C /usr/local --strip-components=1 --no-same-owner \
    && corepack enable

USER ubuntu
~~~

The base image already has the `ubuntu` user that `"user": "ubuntu"` names, and Node lands in `/usr/local/bin`, so `install` finds it without any rc file.

### Idempotent install script

~~~bash
#!/usr/bin/env bash
set -euo pipefail

corepack pnpm install --frozen-lockfile
corepack pnpm generate
~~~

This Node example assumes `package.json` pins pnpm with `packageManager` and defines a deterministic `generate` script. Adapt the image and commands to the repository. Prefer its pinned package manager and lockfile; avoid broad upgrades or lockfile rewrites during environment setup unless the user explicitly requests them.

## Choose the workflow

Read only the reference that matches the request:

- **Create or fully set up an environment:** [Set up a new environment](references/create-environment.md). When build tools are available, that reference includes snapshot → build with `environmentJson` snapshot + install/start → fresh-agent verify → propose with `buildId` before the user Saves.
- **Update a repository-managed environment:** after environment-info confirms a non-empty `environmentJsonPath`, use [Update a repository-managed environment](references/update-repo-managed-environment.md).
- **Update a DB-managed environment:** after environment-info confirms a null or absent `environmentJsonPath`, use [Update a DB-managed environment](references/update-db-managed-environment.md).
- **Verify the current environment's builds:** use [Verify an environment's builds](references/migrate-to-builds.md).

Do not infer the backing type merely from a local file. Use environment-info when available because it reports the effective source used by this run. If the tool is unavailable, inspect the repository, explain what cannot be confirmed, and avoid claiming that a dashboard or repository setting was changed.

For an explanation or audit, the references are usually unnecessary. Read the public setup documentation and current schema when field-level accuracy matters, inspect the effective configuration and referenced files, and explain the lifecycle and precedence directly.


### Build-tested setup and update flows

Follow the matching reference below for whether and when to call `trigger-environment-build`. Do not trigger speculative rebuilds for unrelated questions.

In the create and repository-managed update references, a user request to set up, change, improve, build, or test the environment through this skill is the explicit build request for the MCP tool—do not wait for a second "please build" phrase. Other references (including DB-managed update and migrate-to-builds) keep their own trigger rules; follow those files rather than this summary when they differ.

After calling environment-info:

- If the user wants to verify that the current environment works with builds, or asks to use the build migration guide, read [Verify an environment's builds](references/migrate-to-builds.md) and follow it.
- A non-empty `environmentJsonPath` means repository-managed. Read [Update a repository-managed environment](references/update-repo-managed-environment.md).
- A null or absent `environmentJsonPath` means DB-managed. Read [Update a DB-managed environment](references/update-db-managed-environment.md).

Update flows require an environment ID. For a greenfield create with no linked environment, follow [Set up a new environment](references/create-environment.md) and pass `environmentJson` with optional `snapshot`, `install`, and/or `start` to `trigger-environment-build` so it can create a personal transitional draft.

## Troubleshooting

Find the earliest failing layer:

1. **Image build**: Dockerfile syntax, unavailable packages, build context, architecture, or registry access.
2. **Provisioning and checkout**: repository access, selected ref, runtime user, disk, or network policy.
3. **Install**: non-idempotent scripts, missing lockfiles, private dependency authentication, or generated-file assumptions.
4. **Start and terminals**: commands that exit unexpectedly, bind the wrong interface or port, or depend on setup that did not complete. Builds never run them, so read `/tmp/cursor/start-user/` in a booted agent rather than build logs.

Use setup or build logs as evidence. Reproduce the smallest safe failing command when possible, then rerun the relevant verification. Treat definitive credential, authorization, quota, or entitlement failures as blockers after confirming them once.

## Safety

- Never put tokens, passwords, private keys, or secret values in `environment.json`, Dockerfiles, committed scripts, logs, or chat output. Use supported environment secrets or build-secret mechanisms.
- Do not deploy, publish, apply infrastructure, or mutate production resources as part of environment setup.
- Keep Dockerfiles and install scripts deterministic, non-interactive, and narrowly scoped.
- Do not weaken network, certificate, or package-integrity controls merely to make setup pass.
- Avoid expensive rebuilds until static checks pass. Trigger a build only when the matching workflow reference says to; those references define what counts as explicit user intent for `trigger-environment-build`. Unrelated questions never do.

## Response

Lead with the outcome. Include only the sections relevant to the request:

- Effective configuration source and lifecycle.
- What was inspected or changed.
- Validation evidence.
- Draft-build and subagent evidence, when testing a build.
- Remaining manual action, uncertainty, or blocker.

Only include dashboard links when `environment-info` returns a non-empty environment `url`. Use a markdown hyperlink whose link text is the complete ID:

- Environment / environment dashboard (when directing the user to review build status): `[<environmentPublicId>](https://cursor.com/dashboard/cloud-agents/environments/e/<environmentPublicId>)`
- Build: `[<buildId>](https://cursor.com/dashboard/cloud-agents/builds/<buildId>)`

Use the returned environment `url` for the environment link; builds belonging to that environment may use their build-detail URLs. If `url` is absent—including for a transitional greenfield draft—do not construct links from IDs. Do not append `#builds` — the builds list lives on the default environment detail page.
