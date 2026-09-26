# Watt Toolkit lifecycle test

**Scope:** can DSH safely and reliably start *and stop* the local Watt Toolkit / Steam++ accelerator on
this Windows machine? No BuildReasonSeg model work was done, and Task 6C was not started.
**Machine-readable record:** `evaluation/watt_lifecycle_test.json` (round 3). Rounds 1 and 2 live in
commits `1b41060` and `061afe4` and in the gitignored `artifacts/watt_lifecycle/round{1,2}_report.json`.

## Result (round 3: tray icon off, minimize-on-startup off)

| Question | Answer |
|---|---|
| Watt Toolkit executable found? | **Yes** — Microsoft Store MSIX `4651ED44255E.47979655102CE`, version **3.1.2025.0**, entry point `Steam++.exe` |
| **Does the main page pop up on launch?** | **Yes** — the `Watt Toolkit` window appears **1663 ms** after launch (1749 ms in the confirmation cycle), is **visible**, **not offscreen**, and is the **foreground** window, showing the **home page** |
| **Does acceleration work?** | **Yes, automatically** — `Steam++.Accelerator.exe` takes `0.0.0.0:443`/`:80` within 18.7 s; `huggingface.co` goes from *unresolvable* to serving real metadata |
| **Does closing the main window exit Watt fully?** | **Yes** — via the real window-close command (`WM_SYSCOMMAND`/`SC_CLOSE`, what **✕**/**Alt+F4** send). A bare `WM_CLOSE` (the .NET `CloseMainWindow()` default) is **ignored** |
| **Processes / ports / hosts restored?** | **Yes, byte-identical** |
| User interaction required? | **No** |
| **Classification** | **`FULL_AUTO_OK`** |

## The setting change, verified read-only

`%LOCALAPPDATA%\Packages\4651ED44255E.47979655102CE_k6txddmbb6c52\LocalState\Settings\GeneralSettings.json`

```json
{ "TrayIcon": false, "MinimizeOnStartup": false, "AutoRunOnStartup": true, "WebProxyMode": "FollowSystem" }
```

`...\LocalState\Plugins\Accelerator\Settings\ProxySettings.json`

```json
{ "ProxyMode": "Hosts", "ProgramStartupRunProxy": true, "UseDoh": true, "ProxyMasterDns": "223.5.5.5",
  "SystemProxyPortId": 26561, "Socks5ProxyEnable": false }
```

`MinimizeOnStartup: false` is the new change; `TrayIcon: false` and `ProgramStartupRunProxy: true` are
carried over. `ProxyMode: Hosts` is why acceleration works by rewriting the hosts file.

## 1. Launch → the main page appears by itself

One command (`Start-Process explorer.exe -ArgumentList 'shell:AppsFolder\…!App'`) and:

* the `Watt Toolkit` window existed after **1663 ms**, `IsOffscreen = false`, `IsWindowVisible = true`,
  and `GetForegroundWindow()` returned that same window handle — so it pops up **and** takes focus;
* exactly **one** top-level window exists (no hidden helper window);
* the screen capture shows the **home page**: the left sidebar with 首页 selected (网络加速 / 账号切换 /
  库存游戏 / 本地令牌 / 游戏工具 / 设置 below it), the welcome banner and the 最新消息 feed — **and the
  app's own toast “加速已启动成功”**, which is the app telling us acceleration started;
* `Steam++.exe` (window owner) + a child `Steam++.exe` + `Steam++.Accelerator.exe`.

Capture: `artifacts/watt_lifecycle/screen_round3_launch.png` (gitignored).

## 2. Acceleration comes up by itself

Within **18.7 s** of launch, `Steam++.Accelerator.exe` owned `0.0.0.0:443` and `0.0.0.0:80`, and the hosts
file grew from **5 bytes** to **6873 bytes / 214 lines** with a `# Steam++ Start … # Steam++ End` block
redirecting `github.com`, `api.github.com`, `raw.githubusercontent.com`, `huggingface.co` (and others) to
`127.0.0.1`. No GUI interaction was needed, and no tray icon exists.

## 3. Connectivity while accelerated

| Probe | Baseline | Accelerated |
|---|---|---|
| `huggingface.co` DNS | **unresolvable** | resolves (via hosts → 127.0.0.1) |
| HF `config.json` (WinHTTP) | — | **200**, 1505 bytes, `model_type=qwen3_vl` |
| HF model API (WinHTTP) | — | **200**, 15509 bytes |
| `git ls-remote …/sam2` | OK | **OK**, HEAD `2b90b9f5…` |
| Python + standard `certifi` | GitHub OK, HF fail | **all four `CERTIFICATE_VERIFY_FAILED`** |

No model weights were downloaded. The `certifi` limitation is unchanged across all three rounds: Watt's
interception root is trusted by the **Windows** store (git/schannel, WinHTTP) but not by the project's
`certifi` bundle. The certificate-injection workaround was **not** reintroduced and no insecure flag was
used.

## 4. Closing the main window

| Attempt | Result |
|---|---|
| `CloseMainWindow()` — .NET posts a bare `WM_CLOSE` | returned `True` but **ignored**: 10 s later procs = 3, listeners on `:443`/`:80` = 2, hosts still 6873 bytes with the Steam++ block, window still present |
| **`PostMessage(hWnd, WM_SYSCOMMAND, SC_CLOSE)`** — exactly what Windows sends when the user clicks **✕** or presses **Alt+F4** | **FULL EXIT**: within 12 s procs = 0, listeners = 0, hosts back to 5 bytes with no Steam++ block, window gone |
| whole cycle repeated once with fresh PIDs | window popped at **1749 ms** and was foreground; acceleration auto-enabled; `SC_CLOSE` gave procs = 0, listeners = 0, hosts back to 5 bytes. All four verdicts **True** |

So: **yes, closing the main window exits Watt completely — but only through the real window-close path.**
The `MinimizeOnStartup` change did not alter this; what matters is that `TrayIcon = false` means there is
no tray to hide into.

## 5. Post-exit verification — back to baseline

| Check | Baseline | After exit |
|---|---|---|
| Watt processes / services | 0 / 0 | **0 / 0** |
| Listeners on 443 / 80 | none | **none** |
| Also swept 26561, 8868, 7890, 10808, 10809 | none | **all free** |
| hosts bytes | `ef bb bf 0d 0a` (sha256 `f01a374e…6295`) | **identical** |
| hosts Steam++ block | absent | **absent** |
| System proxy (`ProxyEnable`) | 0 | **0** (nothing left configured) |
| `huggingface.co` DNS | unresolvable | **unresolvable again** |

`github.com` resolves again to `20.205.243.166`, the address that intermittently black-holes TCP 443 on
this network — the pre-existing behaviour already recorded in Task 6B, not a Watt leftover. The hosts/DNS
state is byte-identical to the pre-test baseline.

## 6. All three rounds side by side

| | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| `TrayIcon` | **true** | false | false |
| `MinimizeOnStartup` | true | true | **false** |
| Window on launch | none (tray only); needed a second activation | none; needed a second activation | **pops up by itself in ~1.7 s, foreground** |
| Acceleration automatic | yes | yes | **yes** |
| Bare `WM_CLOSE` (`CloseMainWindow()`) | hid the window to tray, accelerator kept running | ignored, window stayed open | ignored, window stayed open |
| `WM_SYSCOMMAND`/`SC_CLOSE` | not tested | full exit | **full exit** |
| Baseline restored | yes (after manual stop) | yes | **yes** |
| Classification | `AUTO_START_MANUAL_STOP` | `FULL_AUTO_OK` | **`FULL_AUTO_OK`** |

`TrayIcon` decides whether a close request hides the window or ends the process; `MinimizeOnStartup`
decides whether the main page is shown at launch. Neither affects automatic acceleration. **Round 3's
settings are the best of the three**: visible main page on launch *and* a fully automatable exit.

A probe counting bug fixed in round 2 is worth repeating here: the hosts reader counted the UTF-8 BOM as
one "active line", so earlier rounds reported "1 active line" on an effectively empty file. The
**raw-bytes SHA-256** is now authoritative, and the pre-test hosts file has always been the empty 5-byte
BOM+CRLF file.

## 7. Classification and the working sequence

**`FULL_AUTO_OK`** — DSH can start Watt Toolkit, acceleration enables itself, connectivity can be
verified, acceleration can be stopped, Watt can be exited and cleanup can be verified, with no user
interaction.

```
Start-Process explorer.exe -ArgumentList 'shell:AppsFolder\4651ED44255E.47979655102CE_k6txddmbb6c52!App'
# the main page pops up by itself; wait until Steam++.Accelerator.exe owns :443/:80
# then: PostMessage(hWnd of the window named 'Watt Toolkit', WM_SYSCOMMAND=0x0112, SC_CLOSE=0xF060, 0)
```

Never used: `taskkill /F`, `Stop-Process -Force`, killing helper processes, editing hosts, registry,
certificate store, PATH or system proxy, or blind coordinate clicking.

## 8. Consequence for BuildReasonSeg

Watt is what mediated `huggingface.co` in Task 6A and produced the `X-Repo-Commit` /
`LocalEntryNotFoundError` symptoms. The lifecycle is now fully automatable, so DSH can start Watt for
GitHub/Hugging Face work and shut it down afterwards **without user help** — but Task 6B's offline posture
(`HF_HUB_OFFLINE=1` from `local_cache/`) remains the cheaper default, since it needs no accelerator and
avoids the certifi-vs-Watt TLS conflict entirely.

Nothing here justifies re-introducing certificate injection, and no model work was performed.
