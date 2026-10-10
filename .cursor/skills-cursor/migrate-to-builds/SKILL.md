---
name: migrate-to-builds
description: >-
  Verify that a Cloud Agent environment works with prebuilt environment builds
  and recommend any required changes. Use when the user wants to test build
  compatibility or follow the Builds page Verify with Agent flow.
environments:
  - cloud
---
# Verify an Environment's Builds

Use this skill when the user wants to verify that the current Cloud Agent environment works with environment builds, or asks to migrate to builds. Builds do not need to be enabled, so do not ask the user to enable or disable them. Successful verification is the end of this task.

## Resources

| Workflow | Reference |
| --- | --- |
| Verify an existing environment's builds | [Verify an environment's builds](references/migrate-to-builds.md) |

## Choose the workflow

Read [Verify an environment's builds](references/migrate-to-builds.md) and follow it. Call `environment-info` first when available so the reference can classify repository-managed vs DB-managed configuration.

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

The first available source wins. Prefer the Cloud Agent environment-info tool when available; otherwise inspect the repository and explain what cannot be confirmed.

## Safety

- Never put tokens, passwords, private keys, or secret values in `environment.json`, Dockerfiles, committed scripts, logs, or chat output. Use supported environment secrets or build-secret mechanisms.
- Do not deploy, publish, apply infrastructure, or mutate production resources as part of environment setup.
- Keep Dockerfiles and install scripts deterministic, non-interactive, and narrowly scoped.
- Do not weaken network, certificate, or package-integrity controls merely to make setup pass.
- Avoid expensive rebuilds until static checks pass. Trigger a build only when the migrate reference says to. Unrelated questions never do.
- Do not ask the user to enable or disable builds, or to promote, activate, merge, or save a draft as the primary next step.

## Response

Lead with the outcome. Include only the sections relevant to the request:

- Effective configuration source and whether it is repository-managed or DB-managed.
- What was inspected or changed.
- Build and fresh-agent validation evidence.
- Remaining manual action only when configuration still needs a save, a merge, or an external blocker.

When mentioning an environment or build ID in chat, use a markdown hyperlink whose link text is the ID — never a bare ID:

- Environment / environment dashboard (when directing the user to review build status): `[<environmentPublicId>](https://cursor.com/dashboard/cloud-agents/environments/e/<environmentPublicId>)`
- Build: `[<buildId>](https://cursor.com/dashboard/cloud-agents/builds/<buildId>)`

Prefer the environment `url` from environment-info when present; otherwise construct the environment link with the format above. Do not append `#builds` — the builds list lives on the default environment detail page.
