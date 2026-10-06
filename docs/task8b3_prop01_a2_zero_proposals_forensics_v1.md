# PROP01 A2 Zero Proposals Forensics V1

Status: IN_PROGRESS — provenance checkpoint; no inference yet in this task.

Task: `PROP01_A2_ZERO_PROPOSALS_ROOT_CAUSE_FORENSICS_V1`.
Starting branch `docs/governance-v1-1-idle-state-semantics`, HEAD
`504268128b7a9b058580b7702c9429b27705c053`, clean working tree verified after the required five-file read.
Task branch: `audit/task8b3-prop01-a2-zero-proposals-forensics-v1`.
Supervisor attachment was copied verbatim and re-read; attachment and installed task SHA256:
`cd270805140da288a30d8ae67953241763b9de51e089c82a2e5f6ad9b8103a80`.

Exact A2: external RC1 `inference/input/A2.png`, RGB uint8, 1024 x 1024, 1,677,040 bytes,
SHA256 `10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f`.
The historical suite lock binds this hash to A2, `largest_to_left_of_to_nearest`, FAILED/E401,
9 tiles, raw 0, merged 0; the existing A2 diagnostic independently agrees.
Runtime-decoded pixels equal a direct Pillow decode with no transform.

Exact positive control A1: external `inference/input/A1.png`, 1,607,301 bytes,
SHA256 `8a4b459d65773a7dfb0ffcc509c26b5d3a7cea23cd759cdadf94cd46be84c227`,
historical raw 133 / merged 52. It will be used only to establish proposal capability.

Existing detector: `model/buildreasonseg_advisor/detector.pt`, 54,480,241 bytes,
SHA256 `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474`, matches model.yaml.
All 135 actual external manifest-listed files match current canonical Git blobs and canonical manifest hashes.
The external manifest itself has eight stale entries: README, pipeline, model_card, runtime_mapping,
inference README, predict, CLI tests, runtime tests. Actual files match current canonical source.
This is recorded as metadata drift; no external file is changed or synchronized.

Reproduce provenance:

```powershell
& '.conda/buildreasonseg-mvp/python.exe' -B scripts/diagnose_prop01_a2_zero_proposals.py --phase provenance
```

Next: frozen `detect_global` inference with passive observations of actual library results, actual
adapter output, compact calls, merge input/output and library input tensor; then one exact A1 control.
The diagnostic changes only process-local observers that forward original arguments/results unchanged.
No product edits, tuning, sweeps, alternative model, alternative input, or external writes are authorized.
Final acceptance remains with the ChatGPT Supervisor.
