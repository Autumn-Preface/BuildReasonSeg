# Task 5.5 — External datasets for BuildReasonSeg

**Deliverable:** decision register over candidate *external* datasets (pretraining/warm-up, comparison, robustness, domain transfer, paper baselines).
**Project-owned dataset (context, not surveyed here):** BuildSpatialReason-v0.1.1 — 25,229 instruction/mask samples derived from the WHU Building Dataset, 512×512 aerial tiles, instance-level connected components as targets, geometry-verifiable spatial relations.
**Research / access date:** 2026-09-26.
**Evidence base:** `docs/research/_raw/datasets.md` (§0–§11) and `docs/research/_raw/rs_reasoning.md` (§1–§10.4) only. No fetching, no code, no other file modified.
**Licence vocabulary:** `verified first-hand` = read directly off a first-party page retrieved on 2026-09-26 · `UNVERIFIED` = no first-party page for the licence was reachable · `UNVERIFIED / DO NOT USE UNTIL RESOLVED` = required verdict string.
**Footprint vocabulary:** every size is an **estimate** unless the raw file marked it verified; estimates are labelled `EST`. No external dataset publishes a verified total download size in any retrieved page.
**Hard rule applied throughout:** an `UNVERIFIED` dataset is **never** recommended for merging into BuildReasonSeg training.

---

## 1. Decision summary

| Dataset | Verdict | One-phrase reason |
|---|---|---|
| **EarthReason** | `APPROVED_FOR_WARMUP` | Licence-clean (Apache-2.0) reasoning supervision in the RS domain; masks are target-*region*, so it warms up the task, not the instance-building target |
| **RefSegRS** | `APPROVED_FOR_COMPARISON` | Licence-clean (CC-BY-4.0) RS referring baseline, but referring-only and only 4,420 triplets |
| **LaSeRS** | `APPROVED_FOR_COMPARISON` | Licence-clean packaging (Apache-2.0) and instance-level supervision, but the **SAMRS-derived portion inherits academic/non-commercial upstream terms** → eval/baseline only, do not merge until that portion is resolved |
| **DRSeg (PixDLM)** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Best scientific match found (instance masks + multi-step spatial reasoning + aerial/UAV), but licence and release are unverified |
| **RISBench** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Own licence unverified **and** upstream DOTA-v2 is academic-only / no-commercial |
| **RRSIS-D** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Own licence unverified; built on DIOR, which is CC BY-NC 4.0 (non-commercial) |
| **ReasonSeg (LISA data)** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Licence unverified; natural-image domain makes it near-useless for aerial building warm-up anyway |
| **LISA++ data** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | No new dataset at all — curates existing generic segmentation samples; inherits LISA's unresolved status |
| **Terra-CoT / TerraScope-Bench** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | 1M reasoning-chain-embedded pixel masks and multi-hop L2-Spatial tasks, but licence unverified and instance/building coverage unverified |
| **GeoSeg-1M / GeoSeg-Bench (UniGeoSeg)** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Largest candidate by volume (590K images / 1.1M triplets) but licence, instance level and building coverage all unverified |
| **BRIGHT 2026 extension** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | ~291,000 instance-level **building** polygons, but **no language/reasoning channel** — wrong task even if licensed |
| **DOTA-v2.0** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Licence verified **restrictive** (academic use only, commercial use prohibited) and its class list contains **no building class** |
| **VRSBench** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | No data licence stated at all; its "CC BY-SA 4.0" is the website-template licence |
| **SAMRS** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | No SAMRS-level licence; its paper states only that DOTA/DIOR/FAIR1M "can be used for academic purposes" |
| **UAVid-RIS** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | HF dataset card exists but host unreachable; ModelScope returns 404; size/licence/content unverified |
| **DIOR** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Licence verified **restrictive** (CC BY-NC 4.0, non-commercial only); also not an instance-building-mask resource |
| **EarthVQA** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Relational reasoning exists, but Mask Label **✗** per SegEarth-R1 Table 1 — VQA, not segmentation; licence also unverified |
| **GRES / PreGRES (LISAt)** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | RS reasoning-segmentation data with masks implied, but licence and release unverified |
| **TerraLogic** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Hierarchical/multi-hop geospatial reasoning, but **no masks** (mask-less by construction) and licence unverified |
| **OVEarth-Bench** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Zero-shot open-vocabulary eval with reasoning + negative queries; licence unverified |
| **GeoSeg-Bench (Lifan Jiang et al.)** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | 810 image–query pairs, training-free diagnostic bench; licence unverified — **do not confuse with UniGeoSeg's GeoSeg-Bench** |
| **UniRS-Instruct** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Instruction-following RS dataset; contents and licence unverified |
| **Urban Socio-Semantic Segmentation with VL Reasoning (ICLR 2026)** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Existence confirmed from a conference index only; licence and mask properties unverified |
| **ISPRS Annals XI-2-2026, 857 — "From Pixels to Semantics…"** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | **The one item that could overturn §3** — building-specific instruction-tuned VLM; body not readable (see §6) |
| **MRSeg (SegLLM)** | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | Natural-image multi-round reasoning-seg dataset with **reference-mask supervision**; licence unverified and domain-orthogonal |

**Counts (self-consistent):** 28 entries — **3 with a licence verified first-hand and usable** (EarthReason, LaSeRS, RefSegRS), **3 with a licence verified but restrictive or absent** (DIOR CC BY-NC 4.0; DOTA-v2 academic-only; VRSBench no data licence), **22 `UNVERIFIED`**. Group membership is reconciled in §4. No `UNVERIFIED` entry is recommended for merging into training anywhere in this document.

---

## 2. Dataset register

| Dataset | Introduced by | Official source | Images / samples | Annotation type | Reasoning or referring | Image domain | Instance-level? | Building class? | Masks downloadable? | Splits | Licence | Licence verification | Approx. storage | Suitable for |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **EarthReason** | SegEarth-R1, arXiv:2504.09644 | IEEE DataPort DOI 10.21227/2gsy-8b70; ModelScope `earth-insights/EarthReason`; HF `earth-insights/EarthReason` | 5,434 images + masks; >30,000 implicit QA pairs; 200 empty-target images | Reasoning masks + textual QA | **Reasoning** (implicit query; Table 1 "Reasoning Query ✓") | Aerial orthophoto + satellite; sources Million-AID (AID) + fMoW; 0.5 m–153 m; 123²–7617² px | **No** — target *region*, not per-object instance | `UNVERIFIED` (28 categories not enumerated in retrieved text) | Yes in principle (mask label ✓); **IEEE DataPort record says "Files have not been uploaded"** — payload is on ModelScope/HF | **Official:** 2,371 train / 1,135 val / 1,928 test | **Apache-2.0** | **verified first-hand** — ModelScope dataset API `Data.License` | ~8–40 GB (`EST`: 5,434 images at up to 7617² px + masks + ~6 QA/image; no stated figure) | Warm-up on RS implicit-instruction → mask; benchmark table; robustness (empty targets, held-out categories); GSD-spread domain stressor |
| **LaSeRS** | SegEarth-R2, arXiv:2512.20013 (CVPR 2026, pp. 13199–13210) | ModelScope `earth-insights/LaSeRS`; GitHub `earth-insights/SegEarth-R2` | 40,396 masks (~50K produced then filtered); 30,830 QA–mask triples; 122 categories | Instruction + answer + mask + box (TI, TA, Seg, BBox) | **Both** — explicit *and* implicit ("Reasoning Requirements" is a design axis) | Aerial/satellite RS; GSD `UNVERIFIED`; masks seeded from SAMRS + SAM grid prompts | **Yes** — "semantic- and instance-level to part-level" | `UNVERIFIED` (122-category list not retrieved) | Intended ("all data and code will be released"); ModelScope record is public and non-gated → full payload `UNVERIFIED` | Official test = **1,900 held-out masks** partitioned over 4 dimensions; train/val counts `UNVERIFIED` | **Apache-2.0** | **verified first-hand** — ModelScope dataset API `Data.License`; **but SAMRS-derived portion inherits upstream academic/non-commercial terms (§4c)** | ~15–60 GB (`EST`: 40,396 masks over SAMRS-derived tiles; no stated figure) | Comparison/baseline; robustness (4 eval dimensions); **not** a merge target until the SAMRS portion is resolved |
| **ReasonSeg (LISA data)** | LISA, arXiv:2308.00692 | GitHub `dvlab-research/LISA` (the URL named in the LISA abstract) — **unreachable**; HF `Ricky06662/ReasonSeg_test` is third-party | "over one thousand image-instruction-mask data samples" (abstract); 1,218 per SegEarth-R1 Table 1 | Reasoning masks + text; sources OpenImages (Seg) + ScanNetV2 (Seg) | **Reasoning** (implicit) | **Natural images** (indoor ScanNet + OpenImages) — not aerial | Class-agnostic binary mask | Natural-image buildings only (indoor/facade) | `UNVERIFIED` (host unreachable) | **239 training samples confirmed first-hand** (abstract); 200 val / 779 test widely reported but `UNVERIFIED` | `UNVERIFIED` — **paper licence CC-BY-NC-SA-4.0 is a *paper* licence and does not transfer to the dataset** | `UNVERIFIED` (host unreachable) | ~1 GB (`EST`; no stated figure) | Literature comparison only (the canonical reasoning-seg paradigm baseline) |
| **LISA++ data** | LISA++, arXiv:2312.17240 | Code repo URL `UNVERIFIED` (LISA++ abstract names none) | **No new dataset** — "achieved by curating the existing samples of generic segmentation datasets" | Adds instance segmentation + Segmentation-in-Dialogue responses on existing data | **Reasoning** (inherited) | Natural images | `UNVERIFIED` | No aerial buildings | `UNVERIFIED` | Inherits LISA/ReasonSeg splits | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` (host unreachable) | Not applicable (no new data) | **Nothing** — cite as a method only |
| **DRSeg (PixDLM)** | PixDLM, arXiv:2604.15670 (CVPR 2026 Highlight, pp. 26165–26175) | GitHub `XIEFOX/PixDLM` (**unreachable**); HF `WhynotHug/DRSeg` (**unreachable**); ModelScope `WhynotHug/DRSeg` **404 first-hand** | 10,000 high-res UAV images; 10,000 **instance** masks (one target instance per image); 10,000 CoT QA pairs | Instance masks + Chain-of-Thought QA (CODrone rot. boxes → SAM2 → ISAT → GPT-5 annotation → human verification) | **Reasoning** — spatial + attribute + scene-level, genuinely multi-step, CoT traces stored | UAV; oblique + nadir; 30 m / 60 m / 100 m; night + mixed illumination; 58.08 % small instances (<2 % area) | **Yes** | `UNVERIFIED` — built on CODrone; "building" appears only inside one illustrative instruction | Announced ("datasets, models, code available at github.com/XIEFOX/PixDLM"); host unreachable → `UNVERIFIED` | **Official:** 3:2:5 train : val : test; 33.33/33.34/33.33 % spatial/attribute/scene; altitude mix 31.44/25.45/43.11 % | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` (both release hosts unreachable) | ~20–60 GB (`EST`: 10,000 "high-resolution" UAV images; no stated figure) | **Would be** the closest scientific analogue (instance masks + spatial reasoning + aerial) — currently **blocked**; strong robustness axis (oblique/viewpoint/illumination) if resolved |
| **RISBench** | CroBIM, arXiv:2410.08613 (ISPRS J. P&RS) | GitHub `HIT-SIRS/CroBIM` — **unreachable**; announced "will be publicly available" | 52,472 image–language–label triplets; 26 classes; 8 attributes; avg expression 14.31 words | Referring expression + pixel mask + oriented box (boxes/text from **VRSBench**; masks semi-automatic SAM + PA-SAM + human verification) | **Referring only, explicit** (SegEarth-R1 marks "Reasoning Query ✗") | Aerial/satellite; source DOTA-v2 + DIOR; 0.1 m–30 m; 512×512 | Yes (one object per expression) | **No building class** — DOTA-v2's enumerated class list contains none | Announced public; release host unreachable → `UNVERIFIED` | **Official:** 26,300 train / 10,013 val / 16,158 test | `UNVERIFIED` — **upstream constraints verified:** DOTA "academic purposes only… any commercial use is prohibited"; DIOR CC BY-NC 4.0 | `UNVERIFIED` for its own terms; upstream terms **verified first-hand** | ~15–40 GB (`EST`: 52,472 × 512² PNG pre-mask; no stated figure) | Comparison/baseline table **only if** licence resolved; academic/non-commercial at best |
| **RRSIS-D** | RMSIN, CVPR 2024 (Liu et al.), arXiv:2312.12470 | GitHub `Lsan2401/RMSIN` — **unreachable** | 17,402 image–caption–mask triplets; 20 categories | Referring masks (+ boxes) | **Referring only, explicit** — "direct visual attributes such as orientation, color, and size" | Aerial/satellite; source RSVGD (VG) + DIOR (OD) | Yes (one target per caption) | Likely none (DIOR/RSVGD lineage); `UNVERIFIED` | In principle yes; not verified (host unreachable) | Reported across papers; **split definition not verified** → `UNVERIFIED` | `UNVERIFIED / DO NOT USE UNTIL RESOLVED`; built on DIOR → **CC BY-NC 4.0 upstream** | `UNVERIFIED` (host unreachable); upstream **verified first-hand** | ~5–15 GB (`EST`; no stated figure) | Mandatory literature baseline entry; robustness/domain-transfer partner if licensed |
| **RefSegRS** (a.k.a. RegSegRS) | RRSIS, Yuan, Mou, Hua, Zhu, IEEE TGRS 2024 | AI4EO/TUM GitLab `ai4eo/reasoning/rrsis`; HF `JessicaYuan/RefSegRS` | 4,420 image–language–label triplets; 14 categories | Referring masks + text | **Referring, explicit** (SegEarth-R1 "Reasoning Query ✗") | Aerial orthophoto; source **SkyScapes**; 512×512 @ 0.13 m (CroBIM) **or** 800² @ 0.5–30 m (SegEarth-R1) — conflicting | `UNVERIFIED` | `UNVERIFIED` (SkyScapes is an aerial set with building classes; mask provenance unverified) | Yes (public GitLab + HF) | Not verified | **CC-BY-4.0** | **verified first-hand** — official GitLab raw README, "## License / CC-BY-4.0" (repo-level; does not separate data from code) | ~1.5–4 GB (`EST`; no stated figure) | Comparison/baseline table; aerial referring reference point; **not** a source of spatial-reasoning supervision |
| **Terra-CoT / TerraScope-Bench** | TerraScope, CVPR 2026 pp. 16712–16722, arXiv:2603.19039 | CVF open access page; project page `shuyansy.github.io/terrascope/`; HF `sy1998/TerraCoT` (**unreachable**) | Terra-CoT: **1,000,000 samples** with pixel-level masks embedded in reasoning chains, "across multiple sources"; TerraScope-Bench: **3,837 expert-verified questions**, 6 sub-tasks | Reasoning-chain-embedded pixel masks + answers | **Reasoning** (multi-step CoT; **L1** basic grounding, **L2-Spatial** cross-entity relation inference, **L2-Semantic** domain knowledge — i.e. genuine multi-hop spatial) | EO optical **and** SAR, multi-temporal; not video | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | Not verified (bench = 6 sub-tasks) | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` (HF unreachable) | `UNVERIFIED` — no stated figure; 1M masked samples implies a very large corpus | Would be an important **multi-hop** comparison if licensed; currently a watch item |
| **GeoSeg-1M / GeoSeg-Bench (UniGeoSeg)** | UniGeoSeg, CVPR 2026, arXiv:2511.23332 | GitHub `MiliLab/UniGeoSeg` — **unreachable** | GeoSeg-1M: 590K images, 117 categories, 1.1M image–mask–instruction triplets | Referring + interactive + **reasoning** instructions (synthesised from multiple public datasets) | **Both** — explicitly synthesises reasoning instructions | Geospatial/aerial RS | `UNVERIFIED` | `UNVERIFIED` — the **117-category list** would settle this and is the single highest-value missing read (§9.4 #12 of the raw file) | Announced released; host unreachable → `UNVERIFIED` | GeoSeg-Bench is the eval bench; split counts `UNVERIFIED` | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` (HF + GitHub unreachable) | `UNVERIFIED` — no stated figure; by volume the largest candidate surveyed | Would be the strongest **warm-up** candidate by volume if licensed and if buildings are in the 117 classes |
| **BRIGHT 2026 extension** | arXiv:2607.22746 | GitHub `ChenHongruixuan/BRIGHT` (**unreachable**); HF `Kullervo/BRIGHT` (**unreachable**) | ~**291,000 buildings** with instance annotations; 16 disaster events; 7 disaster types; 3 damage labels | **Instance-level building polygons + 3 damage labels**; pre-event optical + post-event SAR | **No language reasoning at all** — supervised damage mapping, not instruction-driven | Satellite; sub-metre pre-event optical + SAR | **Yes** | **Yes — buildings are literally the target class** | Announced public; host unreachable → `UNVERIFIED` | Final phase evaluated on 2 held-out 2025 events | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` (HF + GitHub unreachable) | `UNVERIFIED` — no stated figure | Only as evidence that instance-level building masks exist publicly **without** language; not usable for this task's supervision |
| **DOTA-v2.0** | DOTA (Xia et al., TGRS 2018 lineage); v2.0 is the version used by RISBench | `captain-whu.github.io/DOTA/dataset.html` — **reachable** | `UNVERIFIED` — no image/instance count in the retrieved page | Oriented bounding boxes (OBB); **not pixel masks** | **No language** | Aerial/satellite; images include Google Earth imagery | No (OBB detection) | **No** — enumerated classes are plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool, container crane, airport, helipad | `UNVERIFIED`; registration/gating `UNVERIFIED` | `UNVERIFIED` (train/val/test not stated in retrieved content) | **Academic use only; commercial use prohibited** ("All images and their associated annotations in DOTA can be used for academic purposes only, but any commercial use is prohibited"; Google Earth terms of use apply) | **verified first-hand** — official DOTA dataset page | `UNVERIFIED` — no stated figure | **Upstream reference only** — it is a RISBench source and a SAMRS source; carries no building class and no language |
| **VRSBench** | NeurIPS 2024 Datasets & Benchmarks | `vrsbench.github.io` — **reachable** | `UNVERIFIED` — no count in retrieved content | Boxes + referring text (the upstream source of RISBench's boxes and expressions) | **No** — grounding/VQA-style, not reasoning segmentation | Aerial/satellite | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | **No data licence stated** — the site's CC BY-SA 4.0 is the **website-template licence only** (negative result, verified first-hand) | **verified first-hand** (that no data licence is stated) | `UNVERIFIED` | **Nothing directly** — cite only as RISBench's box/text provenance |
| **SAMRS** | NeurIPS 2023 Datasets & Benchmarks, arXiv:2305.02034 | GitHub `ViTAE-Transformer/SAMRS` (**unreachable**) | `UNVERIFIED`; supplies **~60** of LaSeRS's 122 categories | Mask annotations transferred from source detection datasets | **No** | Aerial/satellite | Not confirmed | `UNVERIFIED` | Announced ("code and dataset will be available"); unverified | `UNVERIFIED` (SAMRS-SOTA ← DOTA-v2.0; SAMRS-SIOR ← DIOR; SAMRS-FAST ← FAIR1M-2.0; + HRSC2016) | `UNVERIFIED` — **no SAMRS-level licence**; the paper states only that "according to the licenses, DOTA, DIOR, and FAIR1M can be used for academic purposes" | `UNVERIFIED` for an SAMRS licence; source-dataset terms stated **in the paper** | `UNVERIFIED` | Only as provenance evidence for LaSeRS's licence constraint |
| **UAVid-RIS** | `UNVERIFIED` (no introducer located) | HF dataset `lironui/UAVid-RIS` (**unreachable**); ModelScope `lironui/UAVid-RIS` → **404 first-hand** | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | UAV / aerial (by name) | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` (host unreachable) | `UNVERIFIED` | **Nothing** until the card is readable |
| **DIOR** | "Object Detection in Optical Remote Sensing Images: A Survey and A New Benchmark", arXiv:1909.00133 | Official author page (Gong Cheng, NWPU) `gcheng-nwpu.github.io` — **reachable** | `UNVERIFIED` in this source (not stated in the retrieved page text) | Object-detection boxes (DIOR / DIOR-R) | **No language** | Aerial/satellite optical | No | `UNVERIFIED` | Not stated | `UNVERIFIED` | **CC BY-NC 4.0** — "Both of the two datasets are freely available under the [CC BY-NC 4.0] license agreement" | **verified first-hand** — official author page | `UNVERIFIED` | **Upstream reference only** — it is the non-commercial constraint behind RRSIS-D, RISBench, VRSBench and SAMRS/LaSeRS |
| **EarthVQA** | AAAI 2024 | `UNVERIFIED` in this source | 6,000 images; 14 classes; 0.3 m; 1024² | VQA annotations — SegEarth-R1 Table 1 shows **Mask Label ✗** | Yes (relational reasoning) | Aerial/satellite | No | `UNVERIFIED` | **No** — VQA, not masks | `UNVERIFIED` | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` | `UNVERIFIED` | Nothing for segmentation; possible related-work citation for relational reasoning on RS imagery |
| **GRES / PreGRES** | LISAt, arXiv:2505.02829 (Quenum et al., UC Berkeley BAIR) | Project page `lisat-bair.github.io/LISAt/`; release URL asserted → `UNVERIFIED` | GRES: 27,615 annotations / 9,205 images; PreGRES: >1M QA | `UNVERIFIED` (reasoning-segmentation VLM datasets) | Reasoning/instruction (task family) | Satellite imagery | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` (host unreachable) | `UNVERIFIED` | Literature comparison only; licence must be resolved first |
| **TerraLogic** | arXiv:2607.12497 (Yan et al.) | GitHub `Ireliya/TerraLogic` — **unreachable** | 545 hierarchy-aware, long-horizon scenario tasks | **No masks** — agent/tool hierarchy (HieraPlan) | Hierarchical/multi-hop geospatial reasoning | Optical, SAR, IR | No | `UNVERIFIED` | **No** | `UNVERIFIED` | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` | `UNVERIFIED` | Evaluation-side prior art for multi-hop geospatial reasoning only |
| **OVEarth-Bench** | arXiv:2607.27278 (Li et al.) | Project page `earth-insights.github.io/OVEarth-bench`; HF `earth-insights/OVEarth-Bench` (**unreachable**) | `UNVERIFIED` | Open-vocabulary queries with mask + box answers, incl. **negative** expressions | Zero-shot evaluation of vocabulary / referring / **reasoning** queries | EO imagery | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | Unified zero-shot protocol; `UNVERIFIED` | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` | `UNVERIFIED` | Possible extra zero-shot eval suite once licensed |
| **GeoSeg-Bench (Lifan Jiang et al.)** | GeoSeg, arXiv:2603.03983 (2026-03-04) | Project page referenced as `tankowa.github.io/GeoSeg.github.io` (`UNVERIFIED`); no repo/weights located | **810 image–query pairs** with hierarchical difficulty levels | `UNVERIFIED` (diagnostic benchmark; method is training-free) | Yes — reasoning-driven, hierarchical difficulty | RS imagery | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` (hierarchical difficulty levels) | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` (paper licence is CC BY 4.0 — **paper only**) | `UNVERIFIED` | `UNVERIFIED` | A 2026 zero-shot comparison if it ever becomes readable. **Naming hazard:** distinct from UniGeoSeg's GeoSeg-Bench |
| **UniRS-Instruct** | IEEE TGRS, "A Principle-Guided Unified Instruction-Following Dataset for Remote Sensing Understanding" | IEEE Xplore `ieeexplore.ieee.org/document/11623679/` (existence only) | `UNVERIFIED` | Instruction-following, "likely mixed" | Instruction-following; reasoning unspecified | RS imagery | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` | `UNVERIFIED` | Nothing until contents are readable |
| **Urban Socio-Semantic Segmentation with Vision-Language Reasoning** | ICLR 2026 (Wang et al.) | `mlanthology.org/iclr/2026/wang2026iclr-urban/` (index only) | `UNVERIFIED` | `UNVERIFIED` (urban socio-semantic, VL reasoning) | Likely reasoning | Urban imagery | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` | `UNVERIFIED` | Watch item only |
| **ISPRS Annals XI-2-2026, 857 — "From Pixels to Semantics: Can a Single Instruction-Tuned VLM Unify Geospatial Building Analysis?"** | ISPRS Annals XI-2-2026, p. 857 | https://isprs-annals.copernicus.org/articles/XI-2-2026/857/2026/ — **reachable but nav-only render** | `UNVERIFIED` | `UNVERIFIED` — title indicates instruction-driven **building** analysis | Likely reasoning (title only) | Geospatial (building analysis) | `UNVERIFIED` | Building analysis is the paper's subject | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` | `UNVERIFIED` (body not retrievable) | `UNVERIFIED` | **Nothing yet** — recorded as the single human-check item in §6 |
| **MRSeg (SegLLM)** | SegLLM, arXiv:2410.18923 (Wang et al., UC Berkeley) | Project page `berkeley-hipie.github.io/segllm.github.io/`; repo `UNVERIFIED (host unreachable)` | `UNVERIFIED` | Multi-round reference + target masks; **reference mask itself supervised** (`L_seg(F([REF],[PAD]), M_ref)`) | **Multi-round conversational reasoning** over prior masks (positional / interactional / hierarchical, 2–8 rounds) | **Natural images** — RefCOCO(+/g), Visual Genome, PACO-LVIS, LVIS, Pascal Panoptic Part, ADE20K, COCO-Stuff | Mixed general instances | No aerial buildings | `UNVERIFIED` | Includes an "MRSeg (hard)" split where previous-round info is necessary | `UNVERIFIED` (dataset licence unreachable; **paper licence is CC BY 4.0** — paper only) | `UNVERIFIED` (host unreachable) | `UNVERIFIED` | Mechanism prior art and a natural-image comparison; **not** usable as RS/building data |

---

## 3. The decisive question

> **Is there any public remote-sensing reasoning-segmentation dataset that combines instance-level BUILDING masks with multi-hop spatial instructions?**

### Answer: **No. As of 2026-09-26 no such dataset could be verified to exist.**

Nothing found satisfies all three conditions at once: (i) public, (ii) instance-level *individual-building* masks, (iii) multi-hop spatial language instructions that require reasoning. The raw evidence partitions cleanly into three failure modes.

**(A) Instance masks exist, but no spatial/reasoning instructions — or no buildings.**
* **LaSeRS** — *instance level: YES.* "approximately 50K high-quality, semantically meaningful masks at varying segmentation granularities, from semantic- and instance-level to part-level" (arXiv HTML 2512.20013v1). *Reasoning: YES*, explicit **and** implicit. *But buildings:* `UNVERIFIED` — the 122 categories are listed only in a supplement that was not retrievable. Fails on the building condition.
* **DRSeg** — *instance level: YES.* "10,000 high-resolution UAV images and 10,000 instance masks", one target instance per image. *Reasoning: YES*, genuinely multi-step (spatial "the vehicle behind the fire truck", attribute, scene-level "the safest landing zone"), with CoT traces stored. *But buildings:* `UNVERIFIED` — it is built on CODrone oriented detection; the word "building" occurs only inside one illustrative instruction ("the building with visible structural damage"), which is not evidence of building coverage. Fails on the building condition.
* **BRIGHT 2026 extension** — *instance level: YES*, ~291,000 buildings with instance polygons from 16 disaster events. *Buildings: YES — literally the target class* ("detect and delineate each building and assign exactly one of three mutually exclusive damage labels"). *Reasoning: NO* — pure supervised damage classification, **no language instructions**. Fails on the reasoning condition.
* **RISBench / RRSIS-D** — one object per expression, i.e. instance-like. *But buildings: NO* — RISBench derives from **DOTA-v2 + DIOR**, and DOTA-v2's enumerated class list (plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool, container crane, airport, helipad) contains **no building class**. *And reasoning: NO* — explicit single referring expressions, marked "Reasoning Query ✗" by SegEarth-R1. Fails on both.

**(B) Reasoning/instructions exist, but the masks are not instance-level building masks.**
* **EarthReason** — *reasoning: YES* (implicit query; the first geospatial pixel-reasoning benchmark). *But instance level: NO* — masks "annotate the target region", annotated from scratch by RS experts with SAM-H as an aid; these are target regions, not per-building instance polygons. *Buildings:* `UNVERIFIED`.
* **ReasonSeg (LISA)** — *reasoning: YES* (implicit, class-agnostic). *But domain: natural images* (OpenImages + ScanNetV2); no aerial imagery, no aerial buildings. Fails on domain.
* **Terra-CoT** — *reasoning: YES*, including **L2-Spatial** multi-hop cross-entity relation inference ("Is the water adjacent to the crops?") composed from L1 questions, with masks interleaved into the trace (1M samples). *But instance level and building coverage:* `UNVERIFIED` — constituent sources and categories were not retrievable. `UNVERIFIED` overall.
* **GeoSeg-1M** — *reasoning: YES*, explicitly synthesises reasoning instructions alongside referring and interactive ones. *But instance level and building coverage:* `UNVERIFIED` — the 117-category list is unread.
* **TerraLogic** — multi-hop/hierarchical geospatial reasoning, but **no masks at all**.

**(C) Neither condition satisfied.**
* **ReasonSeg/LISA++** — natural images; LISA++ releases no new dataset at all ("curating the existing samples of generic segmentation datasets … without … additional data sources").
* **EarthVQA** — relational reasoning yes, but Mask Label **✗** (VQA, not masks).
* **DOTA-v2 / DIOR / VRSBench / SAMRS** — no language-reasoning supervision; DOTA-v2 additionally has no building class.
* **UAVid-RIS, UniRS-Instruct, Urban Socio-Semantic, the ISPRS Annals building paper** — `UNVERIFIED` entirely; nothing can be asserted.

### Evidence summary table

| Candidate | Instance-level? | Buildings specifically? | Multi-hop spatial instructions? | Verdict |
|---|---|---|---|---|
| **LaSeRS** | **Yes** — semantic/instance/part-level | `UNVERIFIED` (122 classes unread) | **Yes** — explicit *and* implicit axis | Fails on buildings (unverified) |
| **DRSeg** | **Yes** — 10,000 instance masks | `UNVERIFIED` (CODrone base; one illustrative mention) | **Yes** — spatial/attribute/scene CoT | Fails on buildings (unverified) |
| **EarthReason** | **No** — target *region* masks | `UNVERIFIED` (28 categories unread) | **Yes** — implicit reasoning | Fails on instance level |
| **RISBench / RRSIS-D** | Yes (one object per expression) | **No** — DOTA-v2 has no building class | **No** — explicit single referring expressions | Fails on both |
| **BRIGHT 2026** | **Yes** — ~291,000 building instance polygons | **Yes — the target class** | **No** — damage labels, no language | Fails on reasoning |
| **ReasonSeg** | class-agnostic binary masks | No aerial buildings (natural images) | Yes | Fails on domain |
| **ReasonSeg source: OpenImages / ScanNetV2** | `UNVERIFIED` | No aerial buildings | n/a | Fails on domain |
| **Terra-CoT / GeoSeg-1M** | `UNVERIFIED` | `UNVERIFIED` | **Yes** | `UNVERIFIED` |
| **TerraLogic** | No masks | `UNVERIFIED` | Yes (hierarchical) | Fails on masks |
| **DOTA-v2 / DIOR / VRSBench / SAMRS** | No (boxes/labels) | No / `UNVERIFIED` | No | Fails on both |

### Consequence for BuildReasonSeg
BuildSpatialReason-v0.1.1 (25,229 instruction/mask samples from the WHU Building Dataset, 512×512 aerial tiles, instance-level connected components, geometry-verifiable spatial relations) occupies a **genuinely unoccupied niche**: instance-level *building* masks + spatial-reasoning instructions on aerial imagery. This is a defensible novelty claim — but it must be stated as **"no public dataset was verified to combine all three"**, not as an absolute. One unread page could overturn it (§6).

---

## 4. Licence status and safe usage

### (a) Licence verified first-hand and usable — 3 datasets

| Dataset | Exact licence | First-hand source of that claim | Conditions |
|---|---|---|---|
| **EarthReason** | **Apache-2.0** | `Data.License` field of the first-party ModelScope dataset API: https://www.modelscope.cn/api/v1/datasets/earth-insights/EarthReason (org `earth-insights`, owner `likyoo` = Kaiyu Li, first author) | Attribution / NOTICE. Not gated on ModelScope. Two caveats: (i) the ModelScope README points users at the **Hugging Face** copy, so re-verify that card before merging *that* copy; (ii) images derive from **Million-AID and fMoW** — verify those upstream terms separately |
| **LaSeRS** | **Apache-2.0** | `Data.License` field of https://www.modelscope.cn/api/v1/datasets/earth-insights/LaSeRS | Attribution. Not gated. **Restricted in substance — see (b) below**, because it imports SAMRS masks. Apache-2.0 covers the LaSeRS *packaging*, not necessarily the upstream masks |
| **RefSegRS** | **CC-BY-4.0** | Official AI4EO/TUM GitLab raw README, "## License / CC-BY-4.0": https://gitlab.lrz.de/ai4eo/reasoning/rrsis/-/raw/main/README.md | Attribution required. CC-BY-4.0 permits derivatives and redistribution. The README does **not** separate data from code — repository-level CC-BY-4.0 is the only statement. Upstream source is **SkyScapes** — verify separately |

### (b) Licence verified but restrictive — 2 datasets (+2 consequences)

| Dataset | Verified restriction | First-hand source | What it forbids / implies |
|---|---|---|---|
| **DIOR** | **CC BY-NC 4.0** — "Both of the two datasets are freely available under the [CC BY-NC 4.0] license agreement" | https://gcheng-nwpu.github.io/ (official author page) | Derivatives and redistribution allowed **for non-commercial purposes only**, with attribution. **Propagates to** RRSIS-D, RISBench, VRSBench and SAMRS — and therefore to the SAMRS-derived portion of LaSeRS |
| **DOTA-v2.0** | **Academic use only; commercial use prohibited** — "All images and their associated annotations in DOTA can be used for academic purposes only, but any commercial use is prohibited"; Google Earth terms of use also apply | https://captain-whu.github.io/DOTA/dataset.html | **Constrains RISBench** (DOTA-v2 is one of its two sources) and SAMRS-SOTA |
| **LaSeRS — the catch** | Its own declaration is Apache-2.0, **but** masks are imported from **SAMRS**, whose sources are DOTA-v2.0 / DIOR / FAIR1M-2.0. The SAMRS paper itself says: "Here, according to the licenses, DOTA, DIOR, and FAIR1M **can be used for academic purposes**." | ModelScope LaSeRS API + https://ar5iv.labs.arxiv.org/html/2305.02034 §2.2 + https://gcheng-nwpu.github.io/ | **A downstream Apache-2.0 declaration cannot enlarge the upstream terms.** Treat the SAMRS-derived portion of LaSeRS as **academic / non-commercial use only**, regardless of the Apache-2.0 label |
| **SAMRS (as an upstream source)** | No SAMRS-level licence was reachable; only the source-dataset statement above | https://ar5iv.labs.arxiv.org/html/2305.02034 | Anything built on SAMRS inherits academic/non-commercial exposure |
| **VRSBench (verified negative result)** | **No data licence stated at all.** The site's "CC BY-SA 4.0" is the **website-template licence** — website code, **not** a dataset licence | https://vrsbench.github.io/ | Do not cite CC BY-SA 4.0 as a VRSBench data licence. RISBench's boxes/text come from VRSBench, so RISBench cannot rely on VRSBench for permission either |

### (c) `UNVERIFIED / DO NOT USE UNTIL RESOLVED` — 22 datasets

RISBench (own terms) · RRSIS-D · DRSeg (PixDLM) · ReasonSeg (LISA data) · LISA++ data · Terra-CoT / TerraScope-Bench · GeoSeg-1M / GeoSeg-Bench (UniGeoSeg) · BRIGHT 2026 extension · UAVid-RIS · EarthVQA · GRES / PreGRES · TerraLogic · OVEarth-Bench · GeoSeg-Bench (Lifan Jiang et al.) · UniRS-Instruct · Urban Socio-Semantic (ICLR 2026) · the ISPRS Annals XI-2-2026 building-VLM benchmark · MRSeg · **SAMRS (own terms)** · **DOTA-v2 (own dataset licence beyond the usage statement)** · **DIOR (own dataset licence verified, but the *dataset as a usable resource* is blocked by non-commercial terms and by having no instance-building masks)** · **VRSBench (as a data resource)**.

**Root cause, identical for all of them:** the only authoritative licence page (GitHub repository `LICENSE`, Hugging Face dataset card, publisher PDF) sits on a host that was DNS-blocked or content-blocked from the research environment on 2026-09-26. Every such URL was attempted once and recorded as unreachable (`datasets.md` §9; `rs_reasoning.md` §8). **No licence value was ever imported from a search snippet, mirror, blog, aggregator or third-party proxy.**

**Explicit warnings.**
1. **Paper licences are not dataset licences.** The LISA paper is CC-BY-NC-SA-4.0, and numerous 2026 papers (TerraScope, UniGeoSeg, BRIGHT 2026, FIRM, Think2Seg-RS, GeoSeg) carry arXiv or CC paper licences. **None of these was used to assign a dataset licence anywhere in this document.**
2. **Data licence ≠ code licence ≠ model-weight licence.** For every dataset here these are three separate licences. The data licences of EarthReason, LaSeRS and RefSegRS are established; the corresponding code licences are **all** `UNVERIFIED (host unreachable)`. Do not assume Apache-2.0 code by inheritance from an Apache-2.0 dataset. Model-weight licences (LISA, PixelLM, Sa2VA, SegEarth-R1/R2, PixDLM, UniGeoSeg) are uniformly `UNVERIFIED`.
3. **The three datasets whose *data* licence is clean are not automatically clean end-to-end:** EarthReason inherits Million-AID/fMoW image terms; LaSeRS inherits SAMRS→DOTA/DIOR/FAIR1M terms; RefSegRS inherits SkyScapes terms. Each needs its own upstream check before a merge.

### Inherited-terms register (datasets carrying upstream licence obligations)

| Dataset | Upstream source carrying terms | Implication |
|---|---|---|
| **LaSeRS** | **SAMRS** (~60 of 122 categories) ← DOTA-v2.0, DIOR, FAIR1M-2.0, HRSC2016 | SAMRS-derived masks: **academic / non-commercial only**; the Apache-2.0 label cannot override this |
| **RISBench** | **DOTA-v2 + DIOR** images; **VRSBench** boxes/text | Academic-only / no-commercial (DOTA) **and** non-commercial (DIOR) **and** no licence at all from VRSBench |
| **RRSIS-D** | **RSVGD + DIOR** | DIOR CC BY-NC 4.0 ⇒ non-commercial only |
| **VRSBench** | None stated | No data licence exists to inherit from — this is itself the problem |
| **SAMRS** | **DOTA-v2.0 / DIOR / FAIR1M-2.0 / HRSC2016** | Academic-purpose statement only |
| **EarthReason** | **Million-AID (AID) + fMoW** image sources | Verify those upstream terms separately; the Apache-2.0 label covers the EarthReason packaging |
| **RefSegRS** | **SkyScapes** | Verify SkyScapes terms separately; CC-BY-4.0 covers the RefSegRS packaging |
| **DIOR** | — (is itself the source of the constraint) | CC BY-NC 4.0, non-commercial |

---

## 5. Recommended use plan

### 5.1 Before BuildSpatialReason (warm-up) — **use nothing external. Use only BuildSpatialReason itself.**
The honest answer is that **no external dataset should be used for warm-up before BuildSpatialReason**, and this is not merely a licensing retreat — it is the scientifically correct call.

* **Of the three licence-clean datasets, none supplies the target supervision.** EarthReason's masks are target *regions*, not per-building instances; RefSegRS is a 4,420-triplet referring set with no spatial-reasoning supervision; LaSeRS does have instance-level masks, but its SAMRS-derived portion carries academic/non-commercial upstream terms and its building coverage is unverified. Warm-up value is therefore **low while licence/preprocessing risk is real** — a poor trade for a project whose entire contribution is *geometry-verifiable instance-level building relations*.
* **The datasets whose semantics would genuinely transfer are all `UNVERIFIED` today:** DRSeg (instance masks + multi-step spatial reasoning, but UAV/CODrone with unverified building coverage and an unreachable licence) and GeoSeg-1M (volume, but instance level and buildings unverified). Under this document's hard rule, neither may be merged.
* **Domain and annotation-convention mismatch is a concrete risk.** EarthReason spans 0.5 m–153 m with 123²–7617² px tiles; BuildSpatialReason is uniform 512×512 aerial tiles with instance connected components. Pretraining on region-mask supervision then fine-tuning on instance masks can teach the wrong mask granularity.
* **Precondition if the project is strictly academic/non-commercial and still wants RS warm-up:** then — and only then — EarthReason (Apache-2.0) may be used as a warm-up set, with the explicit understanding that no public instance-building-relation supervision is available from it. Everything downstream of that decision must remain academic/non-commercial because the *evaluation* partners (RISBench, RRSIS-D) are non-commercial by inheritance anyway.

### 5.2 Comparison / baseline table — permitted, with conditions
| Priority | Dataset | Reason | Licence precondition |
|---|---|---|---|
| 1 | **EarthReason** | The incumbent RS reasoning-segmentation benchmark; every 2025–2026 RS paper reports it (SegEarth-R1/R2, Think2Seg-RS, FIRM, RemoteReasoner) | Apache-2.0 verified. Zero-shot **comparison on a published leaderboard is defensible without redistributing**; re-verify the HF copy before any local use |
| 2 | **RefSegRS** | Small but licence-clean aerial referring set; used by SegEarth-R1 (5,400 training steps) and SegEarth-R2 | CC-BY-4.0 verified; attribution required |
| 3 | **RRSIS-D** | The most-cited RS referring benchmark — mandatory baseline row | `UNVERIFIED` + DIOR non-commercial inheritance. **Cite published numbers; download only after the licence is read first-hand** |
| 3 | **RISBench** | 52,472 triplets; used by SegEarth-R2, Think2Seg-RS, FIRM | `UNVERIFIED` + DOTA-v2 academic-only. Same treatment: cite, do not download |
| 4 | **LaSeRS** | Current SOTA table (FIRM reports 70.5/80.5 gIoU/cIoU); 40,396 masks | Apache-2.0 verified for the LaSeRS packaging; **SAMRS-derived portion is academic/non-commercial** — evaluation use only, no merging |
| 5 | **DRSeg** | Scientifically the closest analogue (instance masks + spatial CoT + aerial) and independently corroborated as a real benchmark by FIRM | `UNVERIFIED` — **cite the paper; do not download** |
| 5 | **MRSeg / ReasonSeg** | Natural-image paradigm baselines used to position the method against generic reasoning segmentation | `UNVERIFIED`; cite only |

### 5.3 Robustness and domain transfer — one usable axis today, two blocked
| Use | Dataset | Reason | Precondition |
|---|---|---|---|
| **Usable now** | **EarthReason** | Contains **200 empty-target images** (tests false-positive behaviour) and deliberately **held-out semantic categories** in val/test (tests open-set generalisation); 0.5–153 m GSD spread is a genuine resolution-transfer stressor | Apache-2.0 verified |
| **Usable now (eval only)** | **LaSeRS** | Test set of **1,900 held-out masks** partitioned across hierarchical granularity (semantic/instance/part), target multiplicity (single/multiple), reasoning requirement (explicit/implicit) and linguistic variability (short/long) — the best-designed robustness grid found | Apache-2.0 verified; SAMRS portion academic/non-commercial ⇒ **evaluation use only** |
| **Blocked, high value** | **DRSeg** | Oblique UAV viewpoints, night/mixed illumination, 30/60/100 m altitudes, 58.08 % small instances — a strong OOD axis for a 512×512 nadir aerial model | `UNVERIFIED` — cannot be used until the HF card and repo licence are read |
| **Blocked** | **Terra-CoT / TerraScope-Bench** | Optical **and SAR**, multi-temporal, multi-hop L2-Spatial questions — the broadest sensor-transfer axis found | `UNVERIFIED` |
| **Blocked, ground-truth-only** | **BRIGHT 2026** | ~291,000 instance-level building polygons — useful as an independent *geometric* reference for building-instance delineation quality even without language | `UNVERIFIED`; note it has no instruction channel, so it cannot serve as a reasoning robustness set |

### 5.4 Literature comparison only (cite, do not download, do not merge)
**DOTA-v2.0, DIOR, VRSBench, SAMRS** (upstream/provenance and prior art; DOTA-v2 has no building class, DIOR is non-commercial, VRSBench states no data licence, SAMRS has no own licence) · **UAVid-RIS, UniRS-Instruct, Urban Socio-Semantic (ICLR 2026)** (existence only) · **TerraLogic** (multi-hop reasoning prior art without masks) · **OVEarth-Bench, GeoSeg-Bench (both senses), GRES/PreGRES** (adjacent evaluation resources, licences unread) · **MRSeg/ReasonSeg** (natural-image paradigm baselines).

### 5.5 Ordered actions
1. **Do not merge any external dataset into BuildSpatialReason training today.** Only EarthReason (and, for academic-only work, the non-SAMRS portion of LaSeRS) has a verified permissive licence, and neither supplies instance-level building masks with spatial relations.
2. **State the novelty claim explicitly as:** "no public remote-sensing dataset was verified to combine instance-level building masks with multi-hop spatial instructions" — with the §6 watch item disclosed.
3. **Build the baseline table from published numbers** for EarthReason, LaSeRS, RefSegRS (+ RRSIS-D and RISBench as cited rows), and cite DRSeg and Terra-CoT as concurrent work.
4. **Before any download**, read the exact licence files listed in `datasets.md` §9.4 (priority order), in particular: the DRSeg HF card (#5/#6), the EarthReason/LaSeRS HF cards (#7/#8) and — highest scientific value — GeoSeg-1M's **117-category list** (#12), which would settle whether buildings are even present.
5. **Human check first:** resolve §6 before finalising the novelty claim.

---

## 6. ISPRS Annals XI-2-2026 building-VLM paper — READ, item CLOSED

| Field | Value |
|---|---|
| **Status** | **READ (abstract + citation metadata) — resolved 2026-09-26 by the task executor** |
| **Exact URL** | https://isprs-annals.copernicus.org/articles/XI-2-2026/857/2026/ |
| **Title** | "From Pixels to Semantics: Can a Single Instruction-Tuned VLM Unify Geospatial Building Analysis?" |
| **Author / venue / date** | Mutreja, Guneet · ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, Vol. XI-2-2026, p. 857 · DOI `10.5194/isprs-annals-XI-2-2026-857-2026` · online **2026-07-03** · verified first-hand from the publisher's citation metadata |
| **How it was read** | The publisher page's `<meta name="citation_*">` tags and embedded abstract were extracted directly from the HTML (155,718 bytes). The earlier "nav-only render" was a fetch-length artefact, not a paywall: the article body is present in the served HTML under the `abstract` meta field. |

**What the paper actually does** (verified from its own abstract):

- Adapts **Google's PaliGemma 2** into a "unified geospatial building analyzer" using a **data-centric**
  methodology.
- Its main contribution is a **pipeline that converts single-modality building polygon annotations into
  a multi-task instruction-tuning dataset of 16,500 samples** spanning **segmentation, detection, VQA
  and captioning**.
- Its three study questions are about single-model multi-task performance, multi-task synergy, and data
  efficiency — **not** about spatial relations.
- Task surface is segmentation / detection / VQA / captioning. There is **no** reference-region token,
  no relation encoder, no relation supervision, no multi-hop spatial chain, and no geometry-verifiable
  relation metric.

**Effect on §3 (the decisive question): NONE. The answer stays NO.**

The paper is a *model/adaptation* study. It does not release, or claim to release, a benchmark that
combines instance-level building masks with multi-hop spatial instructions. The closest overlap is that
it converts building polygon annotations into an instruction-tuning dataset — the same *data-centric
move* as BuildSpatialReason — but its instruction space is multi-task (segment/detect/VQA/caption),
whereas BuildSpatialReason's is relational and geometry-verifiable.

**Effect elsewhere:**

| Where | Effect |
|---|---|
| §3 decisive question | **unchanged — NO** |
| Novelty audit | **PARTIAL_OVERLAP** on "building/remote-sensing specialisation": instruction-tuning a VLM for building analysis is now published prior art (July 2026). The claim "we are the first to instruction-tune a VLM for buildings" must be **dropped**. |
| Novelty audit | It reinforces that the *dataset* contribution must be stated narrowly — relation-grounded, reference-anchored, geometry-verifiable, multi-hop — not "building instructions". |
| §1 / §2 | It is **not** a dataset in the register; no dataset entry is added and none is affected. |
| §4 | No licence claim in this document rested on it; nothing changes. |
| Related-work list | It must be **cited** as the nearest building-specific instruction-tuning work, and used to sharpen the contribution statement. |

**Residual caveat.** Only the abstract and citation metadata were read; the full text was not. If the
full text were to describe a released dataset with instance-level building masks plus multi-hop
spatial instructions, §3 would need revisiting. The abstract's explicit enumeration of its dataset's
task surface (segmentation, detection, VQA, captioning) makes that unlikely, and this is stated as a
reasoned assessment rather than a certainty.

---

### Provenance note
All facts above are traceable to `docs/research/_raw/datasets.md` (§0–§11) and `docs/research/_raw/rs_reasoning.md` (§1–§10.4), access date 2026-09-26, except §6, whose ISPRS citation metadata and abstract were read first-hand by the task executor from the publisher HTML on the same date. No size, licence, split count or URL was invented: every cell that the raw evidence does not support is written `UNVERIFIED`. Repository-content reads obtained by the raw researchers through the third-party `ghproxy.net` proxy were used there **only for dataset-usage facts and never as licence evidence**, and are treated the same way here.

