# Watt Toolkit lifecycle test

**Scope:** can DSH safely and reliably start *and stop* the local Watt Toolkit / Steam++ accelerator on
this Windows machine? No BuildReasonSeg model work was done, and Task 6C was not started.
**Machine-readable record:** `evaluation/watt_lifecycle_test.json` (round 2; round 1 lives in commit
`1b41060` and in the gitignored `artifacts/watt_lifecycle/round1_report.json`)

## Result (round 2, tray icon disabled)

| Question | Answer |
|---|---|
| Watt Toolkit executable found? | **Yes** — Microsoft Store MSIX `4651ED44255E.47979655102CE`, version **3.1.2025.0**, entry point `Steam++.exe` |
| Application start automated? | **Yes** — normal packaged-app activation; process tree in ~22 s |
| Acceleration start automated? | **Yes, automatically** — no GUI interaction at all |
| **Acceleration works?** | **Yes** — `github.com`, `api.github.com`, `raw.githubusercontent.com` and `huggingface.co` become reachable; `huggingface.co` goes from *unresolvable* to serving real metadata |
| **Does closing the main window exit Watt fully?** | **Yes** — via the real window-close command (`WM_SYSCOMMAND`/`SC_CLOSE`, what the **✕** button and **Alt+F4** send). A bare `WM_CLOSE` (the .NET `CloseMainWindow()` default) is **ignored** |
| **Processes / ports / hosts restored?** | **Yes, byte-identical** |
| User interaction required? | **No** |
| **Classification** | **`FULL_AUTO_OK`** |

## The setting change, verified read-only

`%LOCALAPPDATA%\Packages\4651ED44255E.47979655102CE_k6txddmbb6c52\LocalState\Settings\GeneralSettings.json`

```json
{ "TrayIcon": false, "MinimizeOnStartup": true, "AutoRunOnStartup": true, "WebProxyMode": "FollowSystem" }
```

`...\LocalState\Plugins\Accelerator\Settings\ProxySettings.json`

```json
{ "ProxyMode": "Hosts", "ProgramStartupRunProxy": true, "UseDoh": true, "ProxyMasterDns": "223.5.5.5",
  "SystemProxyPortId": 26561, "Socks5ProxyEnable": false }
```

So `TrayIcon: false` is confirmed, `ProgramStartupRunProxy: true` explains the automatic acceleration, and
`ProxyMode: Hosts` explains the hosts-file rewrite.

## 1. Start → acceleration comes up by itself

Launching the packaged app (`Start-Process explorer.exe -ArgumentList 'shell:AppsFolder\…!App'`) produced,
within 22 s:

* `Steam++.exe` (window owner) + a child `Steam++.exe` + `Steam++.Accelerator.exe`;
* `Steam++.Accelerator.exe` owning **0.0.0.0:443** and **0.0.0.0:80**;
* hosts file 5 bytes → **6873 bytes / 214 lines** with a `# Steam++ Start … # Steam++ End` block
  redirecting `github.com`, `api.github.com`, `raw.githubusercontent.com`, `huggingface.co` (and others)
  to `127.0.0.1`.

Because `MinimizeOnStartup: true` **and** the tray icon is off, the first activation comes up with **no
visible window and no tray icon**. A second activation surfaces the window. That is a normal user action
and DSH performs it automatically.

## 2. Connectivity while accelerated

| Probe | Baseline | Accelerated |
|---|---|---|
| `huggingface.co` DNS | **unresolvable** | resolves (via hosts → 127.0.0.1) |
| HF `config.json` (WinHTTP) | — | **200**, 1505 bytes, `model_type=qwen3_vl` |
| HF model API (WinHTTP) | — | **200**, 15509 bytes |
| `git ls-remote …/sam2` | OK | **OK**, HEAD `2b90b9f5…` |
| Python + standard `certifi` | GitHub OK, HF fail | **all four `CERTIFICATE_VERIFY_FAILED`** |

No model weights were downloaded. The `certifi` limitation is unchanged from round 1: Watt's interception
root is trusted by the **Windows** store (git/schannel, WinHTTP) but not by the project's `certifi`
bundle. The certificate-injection workaround was **not** reintroduced and no insecure flag was used.

## 3. Closing the main window — the round-2 headline

| Attempt | Result |
|---|---|
| `CloseMainWindow()` — .NET posts a bare `WM_CLOSE` | returned `True` but **ignored**: 25 s later the window was still present and on-screen, all 3 processes alive, `:443`/`:80` still held, hosts unchanged |
| `CloseMainWindow()` again | ignored again |
| **`PostMessage(hWnd, WM_SYSCOMMAND, SC_CLOSE)`** — exactly what Windows sends when the user clicks **✕** or presses **Alt+F4** | **FULL EXIT**: within 8 s all `Steam++.exe` / `Steam++.Accelerator.exe` processes were gone, the window no longer existed, ports released, hosts restored |
| whole cycle repeated once with fresh PIDs | auto-acceleration **True**, full exit **True**, hosts restored **True** |

So the answer to the question is **yes — but only through the real window-close path**. This also explains
round 1: with `TrayIcon: true` the app treated the close as "hide to tray" (the window vanished while the
accelerator kept running) and the only full-exit route was a tray menu that UI Automation cannot read.
With the tray icon off there is nothing to hide into, and the real close command terminates the whole
process tree.

## 4. Post-exit verification — back to baseline

| Check | Baseline | After exit |
|---|---|---|
| Watt processes | 0 | **0** |
| Watt services | 0 | **0** |
| Listeners on 443 / 80 | none | **none** |
| Also swept 26561, 8868, 7890, 10808, 10809 | none | **all free** |
| hosts bytes | `ef bb bf 0d 0a` (sha256 `f01a374e…6295`) | **identical** |
| hosts Steam++ block | absent | **absent** |
| System proxy (`ProxyEnable`) | 0 | **0** (nothing left configured) |
| `huggingface.co` DNS | unresolvable | **unresolvable again** |

`github.com` again resolves to `20.205.243.166`, which black-holes TCP 443 (~21 s) while `140.82.113.4`
and `20.205.243.168` answer immediately — five consecutive `git ls-remote` attempts failed on that
address. This is the **pre-existing intermittency recorded in Task 6B**, not a Watt leftover: the DNS and
hosts state is identical to baseline.

### A counting bug this round found and fixed

The probe's `total_active_lines` read the hosts file with `Path.read_text`, which normalises newlines and
keeps the UTF-8 BOM. The BOM was therefore counted as one "active line", which is why earlier rounds
reported "1 active line" on an effectively empty file. The probe now strips the BOM and treats the
**raw-bytes SHA-256** as authoritative. The pre-test hosts file has always been the empty 5-byte
BOM+CRLF file; no pre-existing entry was ever lost by Watt.

## 5. Classification

**`FULL_AUTO_OK`** — DSH can start Watt Toolkit, acceleration enables itself, connectivity can be
verified, acceleration can be stopped, Watt can be exited and cleanup can be verified, all with no user
interaction.

Working sequence (recorded verbatim in the JSON):

```
Start-Process explorer.exe -ArgumentList 'shell:AppsFolder\4651ED44255E.47979655102CE_k6txddmbb6c52!App'   # start
# same command again -> surfaces the main window (MinimizeOnStartup + no tray icon)
# then: PostMessage(hWnd of the window named 'Watt Toolkit', WM_SYSCOMMAND=0x0112, SC_CLOSE=0xF060, 0) -> full exit
```

Never used: `taskkill /F`, `Stop-Process -Force`, killing helper processes, editing hosts, registry,
certificate store, PATH or system proxy, or blind coordinate clicking.

## 6. Consequence for BuildReasonSeg

Watt is what mediated `huggingface.co` in Task 6A and produced the `X-Repo-Commit` /
`LocalEntryNotFoundError` symptoms. With the tray icon disabled the whole lifecycle is automatable, so
DSH can start Watt for GitHub/Hugging Face work and shut it down afterwards **without user help** — but
Task 6B's offline posture (`HF_HUB_OFFLINE=1` from `local_cache/`) remains the cheaper default, since it
needs no accelerator and avoids the certifi-vs-Watt TLS conflict entirely.

Nothing here justifies re-introducing certificate injection, and no model work was performed.
