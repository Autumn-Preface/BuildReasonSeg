# BuildReasonSeg Advisor RC1 canonical source

This directory is the Git-tracked canonical source/config layer for the external
BuildReasonSeg Advisor RC1 delivery.

Registered by Task 8B.2-R1 from:
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1

This directory intentionally excludes:
- model weights/checkpoints;
- downloaded Qwen/SAM2 assets;
- user inference images;
- generated masks/overlays/diagnostics;
- logs/caches;
- Conda environments.

The external delivery remains the runnable local package. Changes to RC1
source/config must be made in this canonical tree first and synchronized to the
external delivery using scripts/sync_advisor_rc1_delivery.py.

Task 8B.2-R1 performs registration only and does not change runtime behaviour.
The Ultralytics/readiness issue remains deferred to Task 8B.2-R2 pending
ChatGPT audit.
