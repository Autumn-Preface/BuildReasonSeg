# PROP01 A2 Channel Contract Causal Test V1

Status: IN_PROGRESS; experiment prepared, zero inference so far.
Task: PROP01_A2_CHANNEL_CONTRACT_CAUSAL_TEST_V1.

Starting branch audit/task8b3-prop01-a2-zero-proposals-forensics-v1, HEAD 59296d25195a656e6a75e303e3cbb43fa9df8b8b, clean working tree verified after the five canonical reads.
Task branch audit/task8b3-prop01-a2-channel-contract-causal-test-v1.
Supervisor task copied verbatim and re-read; SHA256 1b99bc499174ba10db15f5cbadf226576a2e0f4983356b11c5afc261225d4c4a.

Exact A2 SHA256 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f; detector SHA256 ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474. Input, model, 135 source files, installed library input-contract source hashes and library versions match the accepted forensic evidence. Nine planned tile hashes match exactly.

Frozen: tile 512 / overlap 128 / stride 384 / imgsz 640 / conf 0.05 / max_det 300 / duplicate IoU 0.50 / mask threshold 0.5 / retina_masks false / CUDA / default NMS / no TTA.

For each tile, first baseline model.predict on existing RGB NumPy, then corrected model.predict on np.ascontiguousarray(RGB_tile[..., ::-1]), with exactly the same arguments and model/backend. Passive preprocessing observer checks baseline tensor equals intended RGB reversed, corrected tensor equals intended RGB. All comparison tensors are diagnostic only. On nonzero baseline or invalid tensor, STOP; no counterfactual interpretation.

Prepared diagnostic: scripts/diagnose_prop01_a2_channel_contract_causal_test.py. --phase prepare performs zero inference; --phase run performs one nine-pair experiment; --phase finalize validates saved evidence with zero inference. Previous diagnostic/evidence remain read-only.

External RC1 is read-only; 2,094-file inventory and 144 protected hashes are unchanged. No product repair, tuning, full pipeline or A1 rerun is authorized. Next gate CHATGPT_PROP01_CHANNEL_CAUSAL_TEST_REMOTE_AUDIT.
