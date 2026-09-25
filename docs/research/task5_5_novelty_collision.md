# Task 5.5 — Novelty-Collision Audit

**Research/access date:** 2026-09-26
**Purpose:** test the BuildReasonSeg contribution claims against 2024–2026 reasoning-segmentation work,
and decide which claims to retain, narrow, reframe, postpone or drop. **No claim below is asserted as
novel on the author's authority alone**; every classification is tied to a source, and the words
"first", "novel" and "state of the art" are used only where a primary source makes that claim.

Classification vocabulary: `DIRECT_OVERLAP` · `PARTIAL_OVERLAP` · `ORTHOGONAL` · `NO_EVIDENCE` · `UNVERIFIED`.

---

## 1. Claims under audit

The proposed causal innovation chain, stated as claims so they can be tested:

1. BuildSpatialReason — a **geometry-verifiable spatial reasoning** supervision dataset.
2. **Reference-guided explicit spatial reasoning**, potentially `[REF] + [SEG]`.
3. **Semantic–spatial–visual fusion**.
4. A **Spatial Relation Encoder** / explicit spatial prior.
5. A **Spatial Consistency Loss**.
6. **Building-group** spatial reasoning segmentation (remote-sensing specialisation).

---

## 2. Collision matrix

`YES` = the work does this · `PARTIAL` = partially, or in a weaker/different form · `NO` = not present ·
`NO_EVIDENCE` = no work found doing this · `?` = could not be established from a reachable source.

| Mechanism ↓ / Work → | LISA | LISA++ | GSVA | PixelLM | GLaMM | PSALM | **SegLLM** | Sa2VA | SAMTok | SegEarth-R1 | SegEarth-R2 | EarthReason | Think2Seg-RS | FIRM | TerraScope | ISPRS building-VLM (2026) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1. Geometry-verifiable spatial reasoning dataset** | NO | NO | NO | NO | NO | NO | PARTIAL | NO | NO | NO | NO | NO | NO | NO | NO | NO |
| **2. `[REF]`-style reference token** | NO | NO | NO | NO | NO | PARTIAL | **YES** | NO | PARTIAL | NO | NO | NO | NO | NO | NO | NO |
| **3. `[SEG]` target token** | **YES** | **YES** | **YES** | **YES** | **YES** | **YES** | **YES** | **YES** | **YES** | ? | ? | ? | NO | ? | ? | NO |
| **4. Reference mask → geometry / spatial prior** | NO | NO | NO | NO | PARTIAL | PARTIAL | **PARTIAL** | NO | PARTIAL | NO | NO | NO | NO | NO | NO | NO |
| **5. Explicit Spatial Relation Encoder** | NO | NO | NO | NO | NO | NO | NO_EVIDENCE | NO | NO | NO | NO | NO | NO | NO | NO | NO |
| **6. Semantic–spatial–visual fusion** | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL | ? | PARTIAL | ? | NO | ? | ? | PARTIAL |
| **7. Spatial consistency loss** | NO | NO | NO | NO | NO | NO | NO | NO | NO | NO | PARTIAL | NO | PARTIAL | NO | NO | NO |
| **8. Multi-hop spatial reasoning supervision** | NO | NO | NO | NO | NO | NO | **PARTIAL** | NO | NO | NO | NO | NO | NO | NO | PARTIAL | NO |
| **9. Building / RS specialisation** | NO | NO | NO | NO | NO | NO | NO | NO | NO | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL | **PARTIAL** |

### 2.1 The cells that matter, with their evidence

- **Row 2 — `[REF]` is taken, twice.** **SegLLM** (arXiv 2410.18923, Oct 2024; authors' project page
  `berkeley-hipie.github.io/segllm.github.io/`; code release `berkeley-hipie/segllm`) generates **two**
  tokens, `[REF]` and `[SEG]`, decoding two masks (`F([REF],[PAD]) → M_ref`, `F([REF],[SEG]) → M_tgt`)
  and supervising **both** with CE + DICE. **PSALM** (ECCV 2024) also uses a `[REF]` token, there as a
  classifier weight. ⇒ **the `[REF]` token idea is not novel; a supervised reference region is not novel.**
  This is the single most damaging finding of the audit.
- **Row 4 — SegLLM already feeds predicted reference geometry back.** Its mask-encoding step computes a
  CLIP embedding of the cropped masked object *and a bounding-box positional embedding derived from the
  predicted mask*, then re-injects both into the LLM input stream for the next round. That is
  "predicted reference region → geometry prior", i.e. **PARTIAL_OVERLAP** with claim 4. The difference is
  the form of the prior: a box/positional embedding consumed by the LLM, versus a *dedicated relation
  encoder* consuming relation-relevant geometry.
- **Row 1 — no relation-level geometric verification found anywhere.** SegLLM supervises a reference
  mask; SegEarth-R2 derives an **attention-space** spatial supervision from GT masks; Think2Seg-RS uses a
  **mask-IoU-only GRPO reward**. None of them re-checks a *stated spatial relation* geometrically against
  the annotation and uses that as the supervision/evaluation signal. This is the strongest surviving
  differentiator.
- **Row 5 — no explicit Spatial Relation Encoder found.** No surveyed work contains a dedicated module
  that takes the geometry of predicted regions and encodes the *relation* between them as a first-class
  representation; relations are either resolved implicitly by the LLM (SegLLM) or omitted.
- **Row 7 — spatial losses exist, but not relation-consistency losses.** SegEarth-R2 adds spatial
  attention supervision; Think2Seg-RS adds a mask-IoU reward. Neither penalises a *relation
  contradiction* (e.g. a target claimed "nearest" that is not the nearest).
- **Row 8 — multi-hop exists as multi-round dialogue.** SegLLM chains rounds ("the object that is
  `<relationship>` the output from round i"), and TerraScope has an L2-Spatial level. Both are
  *conversational/serial* multi-hop rather than a single instruction containing a chained spatial
  relation, and neither supervises the chain with a geometric check.
- **Row 9 — building specialisation is now published prior art.** The ISPRS Annals XI-2-2026 paper
  (Mutreja, DOI `10.5194/isprs-annals-XI-2-2026-857-2026`, online 2026-07-03; abstract read first-hand)
  adapts PaliGemma 2 into a "unified geospatial building analyzer" and, as its main contribution,
  converts **building polygon annotations into a 16,500-sample multi-task instruction-tuning dataset**
  (segmentation, detection, VQA, captioning). It does not do relations, references, multi-hop or
  consistency — but it does occupy "instruction-tuned VLM for buildings".

---

## 3. Verdict per claim

| # | Claim | Verdict | Action |
|---|---|---|---|
| 1 | Geometry-verifiable spatial reasoning dataset | **PARTIAL_OVERLAP** with the general "reasoning-segmentation dataset" line (ReasonSeg/LISA++, LaSeRS, DRSeg, EarthReason), **NO_EVIDENCE** for relation-level geometric verification | **RETAIN, SHARPEN** |
| 2 | Reference-guided `[REF] + [SEG]` | **DIRECT_OVERLAP** on the token (SegLLM; PSALM) | **NARROW and REFRAME** |
| 3 | Semantic–spatial–visual fusion | **PARTIAL_OVERLAP**, very crowded | **REFRAME** |
| 4 | Spatial Relation Encoder | **NO_EVIDENCE** | **RETAIN as the lead contribution** |
| 5 | Spatial Consistency Loss | **NO_EVIDENCE** for a relation-consistency loss; PARTIAL for spatial losses generally | **RETAIN, but define precisely** |
| 6 | Building-group spatial reasoning segmentation | **PARTIAL_OVERLAP** (ISPRS 2026 building-VLM; BRIGHT building polygons; RS reasoning methods) | **RETAIN as the setting, not as a novelty claim** |

---

## 4. Actions

### 4.1 RETAIN (and state carefully)
1. **Geometry-verifiable relation supervision.** The dataset's defining property: every stated relation
   is a frozen predicate that can be re-evaluated geometrically from the annotation, and the v0.1.1
   audit proves it for all 25,229 samples with an independent oracle. Phrase as: *"to our knowledge no
   surveyed public dataset supervises spatial relations in a form that can be re-checked geometrically
   from the annotation"* — an evidenced absence, not an absolute.
2. **An explicit Spatial Relation Encoder over self-predicted region geometry.** No prior work found.
   This should become the paper's lead technical contribution.
3. **A Spatial Consistency Loss on spatial relations** — penalising direction / size / nearest
   contradictions between the predicted target and the predicted reference. No prior work found.
4. **The instance-level building + reference-anchored + multi-hop instruction setting** as the task, with
   the honest statement that no public RS dataset was verified to combine all three.

### 4.2 NARROW
5. **`[REF] + [SEG]`**: from "we introduce a reference token" to *"we consume the reference pathway
   geometrically"*. The token is prior art (SegLLM, PSALM); what is not prior art is **the reference's
   predicted geometry being encoded as an explicit spatial prior by a relation encoder, with a
   consistency loss on the relation**. The MVP's `[REF]` is therefore an *enabler*, never a contribution.

### 4.3 REFRAME
6. **Semantic–spatial–visual fusion**: report it as an architectural detail and an ablation, never as a
   headline claim. Every surveyed pixel-level MLLM fuses language and visual features.
7. **Building specialisation**: move from the contribution list to the *problem setting*. "We are the
   first to instruction-tune a VLM for building analysis" is **false as of July 2026** and must not
   appear.

### 4.4 POSTPONE
8. **Tokenised mask representation** (SAMTok's 2-tokens-per-mask; FIRM's fine-grained intra-token
   representation) — genuinely relevant to small and adjacent buildings, but it replaces the decoder and
   is out of scope before the base pipeline works. Related work + future work.
9. **Rethinking the summarised "reference mask → CLIP embedding" route** — SegLLM already does this; if
   the MVP's `[REF]` ends up as a CLIP-style embedding, the claim collapses to prior art and should be
   dropped rather than defended.

### 4.5 DROP
10. Any "first"/"novel" language about the `[SEG]` token, the `[REF]` token, pixel-level MLLM
    segmentation, or instruction-tuning for buildings.
11. Any claim that the hidden-eligibility correction (Task 5B) is a scientific contribution — it is a
    data-quality fix.
12. Any claim that a Chinese-instruction setting is itself novel; no Chinese-instruction RS
    reasoning-segmentation benchmark was found, but "not found" is not evidence of novelty for a
    language choice, and it is not defensible as a headline claim.

---

## 5. Which result most threatens the project? (reply to Task 5.5 §15 Q11)

**SegLLM (arXiv 2410.18923, "SegLLM: Multi-round Reasoning Segmentation", Oct 2024)** is the single most
threatening prior work, for three separate reasons: it introduces **both** `[REF]` and `[SEG]` tokens
with supervision on both masks; it **re-injects predicted-region geometry** (a mask-derived bounding-box
positional embedding plus a masked-object CLIP embedding) into the LLM, which is a weaker but real form
of the "reference → spatial prior" idea; and it performs **multi-round relational reasoning** over
previously segmented regions. A reviewer who knows SegLLM will read any claim of the form
"we introduce a reference token so the model can reason about a target relative to a reference" as
already published.

Second: **PSALM** (ECCV 2024), whose `[REF]` token shows the token itself was already generic.
Third: the **ISPRS Annals XI-2-2026** building-VLM paper, which closes the "instruction-tuned VLM for
geospatial buildings" framing.

**The defence is narrow but real:** none of these works has a *dedicated relation encoder over predicted
region geometry*, a *relation-level consistency loss*, or *geometry-verifiable relation supervision and
evaluation*. The paper's contribution statement must be built on exactly those three, with `[REF]`
presented as an implementation detail and SegLLM cited explicitly as the closest prior art.

---

## 6. How the Challenge Cup narrative should change (reply to Task 5.5 §15 Q12)

**Before (implied by the current project description):** the story is roughly "we build a
reference-token reasoning-segmentation model for buildings with a spatial fusion module".

**After:** the story becomes a **verification-first** story, in three moves:

1. **The dataset is the foundation, and its property is *checkability*.** Every instruction in
   BuildSpatialReason is backed by a frozen geometric predicate, verified for all 25,229 samples by an
   independent implementation, with zero semantic violations. The talk is not "we made a dataset" but
   "we made spatial supervision that can be *proved* correct" — and the method by which it was proved
   (the independent oracle, the artifact gate) is itself a differentiator in a field where annotation
   quality is rarely auditable.
2. **The model contribution is the explicit relation pathway**, not the tokens: predicted reference
   geometry → Spatial Relation Encoder → semantic–spatial–visual fusion → masked target, with a Spatial
   Consistency Loss that penalises relation contradictions. `[REF]`/`[SEG]` appear as plumbing.
3. **The claim is a measurable one**: on the nontrivial Level-3 subset, the explicit-relation model
   should beat (a) a `[SEG]`-only model, (b) a structured-prompting baseline (Think2Seg-RS-style) and
   (c) a published-style referring-segmentation baseline — evaluated with *relation satisfaction rate*
   and *spatial consistency*, metrics that only a geometry-verifiable dataset makes available.

This reframing is stronger, not weaker: it replaces a crowded mechanism claim with an evidenced
verification claim, and it aligns the technical narrative with what the repository has actually
established (ADR-010, ADR-011, and the frozen relation layer).
