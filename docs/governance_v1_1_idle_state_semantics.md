# Governance V1.1 Idle State Semantics

Task: `GOVERNANCE_V1_1_IDLE_STATE_SEMANTICS`
Status: `COMPLETE` (executor implementation and governance checks; Supervisor remote audit pending).

The Supervisor explicitly authorized this L2 governance correction. The complete supplied
contract was copied verbatim into CURRENT_TASK and reread before implementation.
Its SHA-256 is `6f0d6c04a005ca27dec7f24978270c8dd857d1357afe24f25abd71e3d58b3f75`.

Starting branch: `docs/governance-v1-r1-project-state-correction`.
Starting HEAD: `90209c128a582f6943264c393ef0f383c7b98c4c`.
Preflight: clean working tree; exact required branch and HEAD.
Task branch: `docs/governance-v1-1-idle-state-semantics`.

Actual Git branch / HEAD / diff remain authoritative. AGENTS now defines handoff Git
fields as observational/checkpoint metadata. Idle handoff has no persisted repository
anchor: branch, head, checkpoint commit and pushed are null. Every executor must resolve
actual branch and HEAD from Git at startup. No fabricated or self-referential final SHA
is persisted. CURRENT_TASK returns to NONE / AWAITING_SUPERVISOR / ANY and explicitly
forbids automatic start of PROP-01, 8B.4 or the final demo.

Governance-only validation: YAML parsing and exact idle-state comparison PASS;
CURRENT_TASK idle metadata and startup-anchor statements PASS; pseudo-SHA absent from
both handoff files PASS; AGENTS Git authority semantics PASS; five-path diff gate PASS.
JSON evidence parses successfully. No pytest, inference, training, dependency installation
or sync was run. Product/test/script/delivery paths, PROJECT_STATE and DECISIONS are
unchanged. Scientific state is unchanged. No external RC1 writes were performed.
The path gate uses tracked diff plus untracked-file inventory, and git diff --check
checks whitespace before staging and commit.

Evidence: `evaluation/governance_v1_1_idle_state_semantics.json`.
Final local and remote HEAD must be resolved after push and reported in the final response;
they are deliberately absent from these files to avoid a self-referential commit.
Final milestone acceptance remains with the Supervisor.

Next gate: `CHATGPT_GOVERNANCE_V1_1_REMOTE_AUDIT`. STOP; no product task is authorized.
