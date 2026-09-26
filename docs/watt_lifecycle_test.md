# Standalone Watt Toolkit lifecycle test

**Date:** 2026-09-26
**Scope:** can DSH safely and reliably start *and stop* the local Watt Toolkit / Steam++ accelerator on
this Windows machine? No BuildReasonSeg model work was done, and Task 6C was not started.
**Machine-readable record:** `evaluation/watt_lifecycle_test.json`

## Result

| Question | Answer |
|---|---|
| Watt Toolkit executable found? | **Yes** — but not where a Win32 search looks. It is a Microsoft Store MSIX package. |
| Application start automated? | **Yes** — normal Start-Menu activation, 3 processes within 20 s. |
| Acceleration start automated? | **Yes, automatically** — it enabled itself on launch, with no GUI interaction. |
| Acceleration stop automated? | **No.** |
| Application exit automated? | **No.** |
| Baseline restored? | **No** — the machine is currently still accelerated. |
| **Classification** | **`AUTO_START_MANUAL_STOP`** (see the note on `FAILED_UNSAFE_STATE` below) |

**The user must stop it manually:** open the Watt Toolkit tray icon (taskbar hidden-icons area), turn
acceleration off, then choose *Exit*. The window **X button only minimizes to tray** — do not rely on it.

## What was found

Watt Toolkit is not installed the way a classic desktop app is:

* no Start Menu or Desktop shortcut (187 shortcuts scanned, none matched);
* no entry under `Program Files`, `%LOCALAPPDATA%\Programs` or the uninstall registry keys;
* no `C:\Windows\Prefetch` evidence;
* but `Get-StartApps` lists **`Watt Toolkit`** with AppID `4651ED44255E.47979655102CE_k6txddmbb6c52!App`.

It is a Store MSIX package:

| | |
|---|---|
| Package | `4651ED44255E.47979655102CE` (family `…_k6txddmbb6c52`), version **3.1.2025.0**, `SignatureKind: Store` |
| Entry point | `Steam++.exe` (`Windows.FullTrustApplication`) under `C:\Program Files\WindowsApps\…_x64__…` |
| Capabilities | `internetClient`, `runFullTrust`, **`allowElevation`** |
| Children at runtime | a second `Steam++.exe` and **`Steam++.Accelerator.exe`** (the accelerator module) |

## Baseline (verified, not assumed)

The user's statement was checked rather than believed: **0** Watt processes, **0** Watt services,
**nothing listening on :443 or :80**, hosts file with **1** active line and no GitHub/Hugging Face
entries. `huggingface.co` did **not** resolve; `github.com`, `api.github.com`,
`raw.githubusercontent.com` and PyPI worked over standard TLS, and `git ls-remote` succeeded.

## With acceleration on

Launching the packaged app was enough — **acceleration came up by itself**:

* processes `Steam++.exe` (21808, 2388) + `Steam++.Accelerator.exe` (12380);
* `Steam++.Accelerator.exe` owning **0.0.0.0:443** and **0.0.0.0:80**;
* hosts file rewritten from 1 to **211** active lines, wrapped in `# Steam++ Start` / `# Steam++ End`,
  pointing `github.com`, `api.github.com`, `raw.githubusercontent.com`, `huggingface.co` and others at
  `127.0.0.1`.

Connectivity (lightweight probes only; no weights downloaded):

| Probe | Baseline | Accelerated |
|---|---|---|
| `huggingface.co` DNS | **unresolvable** | resolves (to 127.0.0.1) |
| HF `config.json` (WinHTTP / OS trust store) | — | **200**, 1505 bytes, `model_type=qwen3_vl` |
| HF model API (WinHTTP) | — | **200**, 15509 bytes |
| `git ls-remote github.com/facebookresearch/sam2` | OK | **OK**, HEAD `2b90b9f5…` |
| Python + standard `certifi` (all four URLs) | GitHub OK, HF fail | **all four `CERTIFICATE_VERIFY_FAILED`** |

**TLS limitation (recorded, not worked around).** Watt intercepts TLS with its own root CA, which it
installs into the **Windows** certificate store that git (schannel) and .NET/WinHTTP trust. The project's
standard **certifi** bundle does not, so Python HTTP clients fail verification while acceleration is on.
Per the task instruction the old certificate-injection workaround was **not** reintroduced, and no
insecure flag was used.

## Why the stop side failed

| Attempt | Result |
|---|---|
| 1. Re-activate the packaged app so a window exists | window appeared (`Avalonia-…`, title `Watt Toolkit`) |
| 2. Read-only UI Automation survey | UIAutomationClient loaded, but the window exposes **0 descendants / 0 named controls** — nothing invokable by name, and navigating the app would need blind coordinate clicks |
| 3. `CloseMainWindow()` on the window owner (identical to clicking the title-bar **X**) | returned `True` but **only minimized to tray**: 15 s later all three processes alive, `:443`/`:80` still listening, hosts unchanged |
| 4. Tray route: opened the hidden-icons flyout by UIA `InvokePattern`, found the icon named `' Watt Toolkit'`, right-clicked at the centre of its **UIA-supplied** rectangle | a context-menu popup appeared (Avalonia), but it exposes **0 menu items** to UIA and was not even visible in a screen capture; choosing *Exit* would require clicking unlabelled coordinates |
| 5. | **Stopped.** No `taskkill /F`, no `Stop-Process -Force`, no helper killed, no hosts/registry/certificate/proxy edit. |

So the lifecycle capability is asymmetric: **start is fully automatic, stop is not automatable** on this
version — the app is a tray-only Avalonia build with no UI Automation tree and no documented CLI shutdown
switch.

## Classification, and the `FAILED_UNSAFE_STATE` question

Primary: **`AUTO_START_MANUAL_STOP`** — "DSH can reliably start Watt/acceleration, but cannot safely stop
and exit it."

The spec also defines `FAILED_UNSAFE_STATE` as "the test leaves a network/proxy state different from
baseline and DSH cannot restore it safely", and **that condition currently holds**: the hosts block is
still in place, the accelerator still owns `:443`/`:80`, and baseline is not restored. It is recorded as a
*secondary condition* rather than the primary classification because nothing is broken — the app is
working exactly as designed and the user can restore baseline in two clicks. Whichever label a reviewer
prefers, the actionable content is the same and is listed under "manual action" below.

## Manual action required now

1. Open the taskbar hidden-icons area and click the **Watt Toolkit** icon (or use its window).
2. Turn acceleration **off**.
3. Choose **Exit** (退出). Do not rely on the window **X** — it only minimizes to tray.
4. Expected baseline afterwards: **no** `Steam++.exe` / `Steam++.Accelerator.exe` processes, **nothing**
   listening on `:443` or `:80`, and the hosts file back to **1** active line
   (sha256 `b42f2099187886def637d6aa840022266e05cb6c987a9394e708e23cd505eb46`).

## Consequence for BuildReasonSeg

Watt was what mediated `huggingface.co` in Task 6A and produced the `X-Repo-Commit` /
`LocalEntryNotFoundError` symptoms. Since its stop side cannot be automated, the workable pattern is:
**DSH starts Watt when networking to GitHub/Hugging Face is needed, and the user stops it afterwards.**
Task 6B's offline posture (`HF_HUB_OFFLINE=1` from `local_cache/`) remains the safer default: it needs no
accelerator at all, and it avoids the certifi-vs-Watt TLS conflict entirely.

Nothing in this test justifies re-introducing certificate injection, and no model work was performed.
