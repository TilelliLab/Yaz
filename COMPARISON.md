# How Yaz compares to other knowledge-editing methods

> **Read this honestly.** Yaz is a ≈807K-param research prototype on 50 synthetic facts. It is **not**
> state-of-the-art, **not** larger-scale, and **not** a unique capability. This page exists so you can
> place it accurately against the literature — including where it loses. Every Yaz number here is
> reproducible from this repo; every cross-method *property* is qualitative and from the public papers
> (cited where we're confident, marked otherwise).

## The three families of knowledge editing

| Family | Idea | Examples |
|---|---|---|
| **Edit the base weights** | Change the model's own parameters so it "knows" the new fact | ROME, MEMIT, MEND, KnowledgeNeurons |
| **Add a side memory / adapter** | Leave the base frozen; route to an external store of edits | SERAC, GRACE, WISE, MELO, PENME, RECIPE |
| **Edit in context** | Put the correction in the prompt | IKE and in-context variants |

Yaz is in the **side-memory** family, with two unusual properties: its edit is a **structural** column
swap (locality holds by construction, not just empirically), and it **abstains** on low routing
confidence instead of answering with the base model.

## Capability matrix (qualitative)

| Method | Edit lives in | Retrain-free edit | Locality | Sequential editing | Abstains when unsure? | Model-agnostic |
|---|---|---|---|---|---|---|
| **ROME** | base FFN weights (rank-1, closed-form) | yes | empirical; known to degrade under many sequential edits | limited before quality loss | no | architecture-specific (GPT-style) |
| **MEMIT** | base FFN weights (many layers) | yes | empirical | scales to many edits, but selectivity erodes | no | architecture-specific |
| **MEND** | base weights via a trained hypernetwork | needs a trained hypernetwork | empirical | limited | no | per-model training |
| **SERAC** | external memory + classifier + counterfactual model | yes | scoped by a learned classifier | many | no (routes to base) | wrapper |
| **GRACE** | added codebook of activations | yes | strong, but keyed on activations (paraphrase-sensitive) | thousands (paper) | no (routes to base) | layer-specific |
| **WISE** | added side memory | yes | strong | designed for long edit streams | no (routes to base) | wrapper |
| **PENME** | embedding-keyed adapter memory | yes | scoped by embedding | many | no | wrapper |
| **Yaz** | its own decoder columns (*atoms*) | yes | **structural** (disjoint columns) | flat to 40 edits (tiny scale) | **yes** (routing margin) | no (intrinsic to Yaz) |

The one column where Yaz stands alone is **abstention**: published editors treat low confidence as
"route to the base model and answer"; Yaz declines ("I'm not sure which fact you mean"). This is a real,
under-occupied feature — but it is a *step*, not a moat: selective prediction + a margin threshold is
textbook, and any of these editors could add it.

## Empirical head-to-head — selectivity vs ROME/MEMIT (our controlled study)

These numbers are from **our own controlled experiments on the same tiny synthetic task** (country→capital,
a ≈256K-param Yaz, ROME/MEMIT applied to the model's unembedding). They are **not** a benchmark of ROME/MEMIT
on production LLMs — they show how the *mechanisms* behave on an identical small task. The qualitative
finding (weight-editing loses selectivity as edits accumulate) is independently well-established in the
literature.

**At 5 facts** — equivalent: Yaz and ROME both ~0/4 side-effects. Weight editing is fine when edits are few.

**At 50 facts** (5 UPDATE edits, side-effects measured against the other 49 facts):

| | ROME | MEMIT (joint) | MEMIT-proper (cov) | **Yaz** |
|---|---|---|---|---|
| Edits hitting target rank-1 | 4/5 | 4/5 | 4/5 | 4/5 (one PARTIAL) |
| Non-edit side-effect rate | 22–67% | 62% | 67% | **0%** |
| Aggregate verdict | DEAD 0/5 | DEAD 0/5 | DEAD 0/5 | **ALIVE 4/5** |

ROME/MEMIT land the new answer **but corrupt a large fraction of the other facts**. Yaz's structural
locality keeps side-effects at 0/49.

**At 200 facts** (selectivity holds, but Yaz's own reliability degrades):

| | Yaz (256K, 200 facts) |
|---|---|
| Max side-effect rate per UPDATE | **0–0.5% (≤1/199)** |
| UPDATE reliability | drops to 1/5 ALIVE, 5/5 PARTIAL — the tiny model can only memorize 101/200 facts |

**Honest read:** Yaz's advantage is **selectivity/locality** (it edits one fact without touching others),
which weight-editing loses at scale. Yaz's **weakness** is reliability and capacity — at 200 facts the
807K/256K model can't hold all the facts, so UPDATE success falls. ROME/MEMIT scale to far more real-world
knowledge than Yaz ever has.

## Yaz's published-model numbers (reproducible here)

The shipped model is the 50-fact `semantic_v2` routing/abstention model (`python scripts/scaling/s3_route_abstain.py`):

- Abstention risk-coverage **AURC 0.004** (oracle 0.003; random 0.087); coverage @ ≤5% risk **96.6%**.
- Paraphrase routing reach **0.696** held-out (vs 0.216 surface routing).
- Per-edit locality **0/10 collateral**, bpc **+0.000%** across 40 sequential edits; retention flat 1.000.
- CREATE passes a 4/4 battery (monosemantic / local / readable / deletable).

## Caveats on this comparison

- **Provenance:** the ROME/MEMIT selectivity numbers come from an earlier Yaz CRUD study (d_model=64,
  ROME-on-unembedding), a *different experimental line* from the shipped routing model. They're real and
  in this project's results, but they are a controlled-task illustration, not a production benchmark.
- **First-byte only:** Yaz edits set the answer's first byte; full-word transfer ≈0.05. The selectivity
  comparison is on first-byte targets.
- **Tiny, synthetic:** 50–200 country→capital facts. None of this is validated on open-vocabulary knowledge.
- **Not a moat:** every mechanism above is published and copyable. Yaz is a clean recombination whose only
  genuine differentiators are sub-1M-param/CPU scale and abstention-as-refusal.

*Cross-method specifics (exact max-edit counts, tool support in EasyEdit, etc.) are intentionally left
qualitative here rather than risk citing an unverified number. See each method's paper for precise figures.*
