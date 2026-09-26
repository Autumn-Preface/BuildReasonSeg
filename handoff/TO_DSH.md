# TO_DSH — Standalone Watt Toolkit Lifecycle Test

> Status: ACTIVE
>
> Repository context: `BuildReasonSeg`
>
> Purpose: only test whether DSH can safely and reliably start and stop Watt Toolkit / Steam++ on this Windows machine.
>
> This task is intentionally separate from BuildReasonSeg model work.
>
> Do not train, modify model architecture, download model weights, alter datasets, or begin Task 6C.

## 0. User-facing language

All narrative text shown to the user in the DSH web/chat UI must be Chinese.

Commands, paths, process names, executable names, ports, registry keys, logs, and field names may remain English.

## 1. Safety boundary

DSH currently has Full Access only because the normal Windows sandbox is unavailable.

Full Access does not authorize unrelated system modification.

Allowed:
- read-only process inspection;
- read-only service inspection;
- read-only registry inspection if needed to locate the installed application;
- read-only shortcut inspection;
- read-only filesystem search limited to likely Watt Toolkit installation/start-menu locations;
- launching the installed Watt Toolkit application;
- interacting with Watt Toolkit through normal application/GUI-safe mechanisms if technically available;
- normal graceful close/exit requests;
- lightweight connectivity checks to GitHub / Hugging Face;
- inspecting hosts file;
- inspecting listening ports;
- inspecting DNS resolution;
- writing test logs/reports only inside `BuildReasonSeg`.

Forbidden:
- editing Windows hosts file;
- editing registry;
- editing certificate stores;
- editing system proxy configuration;
- editing PATH;
- installing/uninstalling/updating Watt Toolkit;
- installing any new software;
- using `taskkill /F`;
- force-killing Watt Toolkit or related processes;
- terminating unrelated processes;
- changing Windows services;
- disabling TLS verification;
- using `verify=False`;
- adding/removing root certificates;
- downloading model weights or datasets;
- modifying BuildReasonSeg model code;
- modifying training configs;
- starting Task 6C.

If graceful shutdown cannot be verified, leave Watt running and ask the user to close it manually.

## 2. Current expected state

At task start, the user says:
- Watt Toolkit acceleration is stopped;
- Watt Toolkit is exited.

Do not assume this blindly; verify it.

## 3. Baseline state capture

Before starting Watt Toolkit, record:

### Process state
Check for processes plausibly related to:
- Watt Toolkit
- Steam++
- SteamTools
- BeyondDimension

Record process name, PID, executable path when available.

### Listening ports
Inspect at minimum:
- TCP 443
- TCP 80
- any localhost ports owned by Watt-related processes if present

### Hosts file
Read only:
`C:\Windows\System32\drivers\etc\hosts`

Record only entries relevant to:
- `github.com`
- `api.github.com`
- `raw.githubusercontent.com`
- `huggingface.co`
- `hf.co`
- clearly Watt-generated GitHub/Hugging Face acceleration entries

Do not modify the file.

### DNS / connectivity baseline
Probe:
- `https://github.com`
- `https://api.github.com`
- `https://raw.githubusercontent.com`
- `https://huggingface.co`

Use DNS resolution, lightweight HTTPS HEAD/GET, and:
`git ls-remote https://github.com/facebookresearch/sam2 HEAD`

Record reachable/unreachable, DNS result, timeout/error, elapsed time.

Write preliminary data to:
`evaluation/watt_lifecycle_test.json`

## 4. Locate Watt Toolkit executable

Locate the installed Watt Toolkit / Steam++ executable without changing anything.

Preferred search order:
1. running-process metadata if unexpectedly already running;
2. Start Menu shortcuts;
3. Desktop shortcuts;
4. common installation directories;
5. read-only uninstall/application registry entries if necessary.

Do not search the entire disk recursively unless necessary.

Record executable path, application/product name, version if readable, and whether a shortcut or direct executable was found.

If executable cannot be found, stop and tell the user what information is needed.

## 5. Automated startup test

Attempt to start Watt Toolkit using the normal executable, e.g. PowerShell `Start-Process`.

Do not use compatibility flags, elevated mode, or undocumented switches unless already present in the user's shortcut.

After launch, wait reasonably and verify:
- expected Watt-related process exists;
- process path matches located installation;
- application did not immediately crash.

Record startup time.

Important: starting Watt Toolkit is not the same as starting acceleration.

Determine whether acceleration becomes active automatically.

Check:
- hosts changes;
- localhost listeners;
- GitHub/Hugging Face connectivity;
- Watt process behavior.

If launching the application alone does not enable acceleration and enabling it requires GUI interaction:
- attempt GUI interaction only if DSH can do so reliably and safely using available Windows interaction capabilities;
- do not guess screen coordinates blindly;
- do not automate unrelated settings.

If DSH cannot reliably operate the GUI control that starts acceleration, stop the automation attempt and tell the user:
“Watt Toolkit 已成功自动启动，但加速需要你手动开启。”

Then wait for the user rather than trying unsafe automation.

## 6. Acceleration-active verification

Once acceleration is active, verify:

### Process / proxy evidence
Record:
- Watt-related process(es);
- relevant local listeners;
- any hosts entries created or changed by Watt.

### Connectivity
Probe again:
- GitHub HTTPS
- GitHub API
- raw.githubusercontent.com
- Hugging Face
- `git ls-remote`

Success criterion:
At least GitHub and Hugging Face connectivity should improve from baseline where they were unavailable or unreliable.

Do not download large files.

### TLS integrity
Use standard TLS verification.

Do not:
- inject SteamTools certificates into project certifi;
- modify certifi;
- use insecure flags.

If standard Python `certifi` does not trust Watt interception again, record it as a limitation. Do not reintroduce the old cert workaround.

## 7. Lightweight network-use simulation

While Watt acceleration is active, perform only:

1. GitHub:
   `git ls-remote https://github.com/facebookresearch/sam2 HEAD`

2. Hugging Face:
   fetch model metadata or a tiny file such as `config.json` / API JSON response.

Do not download model weights.

Record success/failure.

## 8. Graceful stop / exit test

Test whether DSH can safely return the machine to the pre-test state.

Preferred shutdown order:
1. stop acceleration;
2. exit application.

Use only normal mechanisms:
- documented application commands if discovered;
- reliable tray/menu/window automation;
- normal application close/exit request.

Do not use:
- `taskkill /F`;
- `Stop-Process -Force`;
- killing helper processes;
- direct hosts edits;
- direct registry edits;
- certificate changes.

Do not treat “main window disappeared” as proof of exit. Watt may minimize to tray.

Verify process state.

If DSH can stop acceleration but cannot exit safely:
- leave app running;
- report `AUTO_STOP_ACCELERATION_ONLY`.

If DSH can launch but cannot safely stop acceleration:
- do not force terminate;
- ask user to manually stop acceleration and exit;
- report `MANUAL_STOP_REQUIRED`.

## 9. Post-shutdown verification

After graceful shutdown attempt, verify:

### Processes
No Watt-related process remains if full exit was expected.

### Ports
No Watt-related local listeners remain.

### Hosts
Compare relevant hosts entries with baseline.

They should return to original baseline state.

Do not edit anything if they do not.

### Connectivity
Re-test GitHub/Hugging Face.

If baseline was unreachable without Watt, it is acceptable for them to become unreachable again.

The important condition is return to baseline rather than leaving a half-configured state.

## 10. Lifecycle classification

Classify exactly one:

### `FULL_AUTO_OK`
DSH can:
- start Watt;
- enable acceleration;
- verify connectivity;
- stop acceleration;
- exit Watt;
- verify cleanup;
without user interaction.

### `AUTO_START_MANUAL_STOP`
DSH can reliably start Watt/acceleration, but cannot safely stop and exit it.

Future workflow:
- DSH starts automatically;
- user manually stops/exits after networking.

### `MANUAL_START_AUTO_STOP`
Startup/acceleration needs user action, but DSH can safely stop/exit.

### `MANUAL_START_STOP`
DSH cannot safely automate either side.

### `AUTO_APP_ONLY`
DSH can start/exit Watt application but cannot reliably toggle acceleration.

### `FAILED_UNSAFE_STATE`
Only if the test leaves a network/proxy state different from baseline and DSH cannot restore it safely.

If this occurs, immediately tell the user what remains and what manual action is needed.

## 11. Required report

Create:
`evaluation/watt_lifecycle_test.json`

Include:
- test timestamp;
- Watt executable path;
- version if available;
- baseline process state;
- baseline relevant hosts entries;
- baseline listeners;
- baseline DNS/connectivity;
- startup method;
- startup success;
- whether acceleration could be enabled automatically;
- active-state processes/listeners/hosts;
- active-state GitHub test;
- active-state Hugging Face test;
- stop-acceleration method;
- exit method;
- post-stop process state;
- post-stop listeners;
- post-stop hosts comparison;
- post-stop connectivity;
- classification;
- user interaction required;
- safety warnings;
- exact commands/actions used.

Also create:
`docs/watt_lifecycle_test.md`

with a concise human-readable explanation.

## 12. Git policy

This task is not model development.

Do not modify existing BuildReasonSeg architecture/training code merely for this test.

Only commit:
- `evaluation/watt_lifecycle_test.json`
- `docs/watt_lifecycle_test.md`
- optional handoff/state updates

Recommended commit:
`test: verify Watt Toolkit lifecycle`

If GitHub is unreachable after Watt is correctly shut down, do not restart Watt solely to push unless automated restart is already proven safe.

Instead:
- keep local commit;
- report commit hash;
- tell user push requires Watt/manual networking.

If Watt remains running as part of a successful test and networking is available, normal push is allowed.

## 13. Final DSH web/chat response — Chinese only

Report:
- lifecycle classification;
- whether Watt executable was found;
- whether app startup was automated;
- whether acceleration startup was automated;
- GitHub/Hugging Face behavior before and during Watt;
- whether acceleration was stopped automatically;
- whether Watt exited automatically;
- whether hosts/listeners returned to baseline;
- whether user intervention is required in future;
- local commit hash;
- push success/failure.

If manual action is needed right now, make that the first sentence.

## 14. STOP

After the Watt lifecycle test:

STOP.

Do not start any BuildReasonSeg model work.
Do not begin Task 6C.

Wait for ChatGPT review and a separate project-task instruction.
