# Task 8B.3-P1D11A-R2 — Supported-Domain Policy Implemented (canonical)

## 1. Task and scope

Policy implementation in canonical RC1 documentation with manifest identities maintained under the frozen
`GIT_CANONICAL_BLOB_BYTES` convention. No model, runtime or test change and **no external write sync**.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 581f91de8145a5b680d510b185b38b6e9878aa1d
pre-edit gate (Git canonical HEAD blobs)  = 135/135 PASS
post-edit gate (Git canonical index blobs) = 135/135 PASS · 135 entries · path list identical
```

## 2. Manifest entries updated (exactly two, canonical identities)

| entry | bytes before | bytes after | sha256 after |
|---|---:|---:|---|
| `README.md` | 9335 | 10411 | `8615537af6448321…` |
| `docs/model_card.md` | 5900 | 6945 | `1c190ad3c260cdd0…` |

`schema` stays `BuildReasonSeg.AdvisorRC1.SourceManifest.v1`, `identity_basis` stays `GIT_CANONICAL_BLOB_BYTES`, no
entry was added or removed, no unrelated identity changed and no per-entry metadata was introduced.

## 3. Policy document created

`docs/task8b3_p1d11_supported_domain_policy.md` — positively verified scope (§5.1), explicit non-claims plus the
preserved scene-disjoint limitation (§5.2), the A2 policy `DOCUMENTED_PERSISTENT_NON_DETECTION` with bounded evidence
only (§5.3), the two distinct Demo sets with the four immutable locked sample IDs (§5.4), and the verbatim English plus
faithful Chinese test-reuse disclosure (§5.5).

## 4. Canonical RC1 documents updated

| document | change |
|---|---|
| `README.md` | `## 输入` distinguishes accepted software input modality from the verified research/evaluation domain; new `### 已验证数据域与 Demo 边界` subsection records the domain, the scene-disjoint caveat, the no-guarantee statement, the A2 policy, the untouched six-case suite, the four locked qualitative candidates and the disclosed deterministic reuse |
| `docs/model_card.md` | `## 2. 数据` states the verified domain explicitly; new `### Demo policy` subsection records the retained historical suite, the metadata-only locked candidates, the disclosure, the non-established robustness claims and the A2 policy |

## 5. Unchanged by this task

```text
runtime / tests / .gitattributes / external delivery = UNCHANGED
model runs / inference                               = NONE
frozen metrics / architecture / locked candidates    = UNCHANGED
PROP-01 status                                       = OPEN (not closed)
```

## 6. Status fields

```text
supported-domain policy     = IMPLEMENTED_IN_CANONICAL_RC1_DOCS
A2 policy status            = DOCUMENTED_PERSISTENT_NON_DETECTION
canonical README policy     = UPDATED
canonical model card policy = UPDATED
canonical manifest          = 135/135 PASS under GIT_CANONICAL_BLOB_BYTES
next gate                   = canonical → external controlled sync (separate task, NOT executed here)
```
