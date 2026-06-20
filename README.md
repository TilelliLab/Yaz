# Yaz

A sub-1M-parameter, byte-level language model whose individual facts you can **create, read,
update, and delete** one at a time — with provable per-edit locality — and that **abstains** when
it isn't confident which fact you mean, instead of guessing. Runs on CPU, offline.

> **Status: research prototype.** Everything here is small-scale and honestly scoped (see
> [Caveats](#caveats)). It is a clean, reproducible demonstration — not a production system and not a
> state-of-the-art result.

## Idea

Each fact lives in its own addressable **atom** (one column of an additive decoder). A prompt is
routed to a fact by a **frozen sentence embedding** (so paraphrases reach the same fact), and the
routed atom contributes the answer. Because facts are disjoint columns:

- **UPDATE** a fact = swap one decoder column (no retraining).
- **DELETE** a fact = zero its atom (others provably untouched).
- **CREATE** a fact = allocate a fresh atom.
- **Locality** is structural: editing fact A cannot change fact B's output (given no routing collision).
- **Abstention**: the routing **confidence margin** (top-1 minus top-2) is a calibrated "I don't know
  which fact you mean" signal — the model refuses low-confidence queries.

## Quick start

```bash
# 1) deps (CPU-only). Use the CPU wheel index so pip doesn't pull the multi-GB CUDA stack:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# (plain `pip install -r requirements.txt` also works but may download a CUDA torch build.)
# On first run, sentence-transformers downloads all-MiniLM-L6-v2 (~90 MB) once; fully
# offline thereafter.

# 2) try the demo (routes a prompt, answers, or abstains; edits/deletes are live)
python demo.py --demo
python demo.py --prompt "the country of the Eiffel Tower, its capital is "
python demo.py --prompt "The capital of France is " --edit France=Lima
python demo.py --prompt "best pizza topping?"        # -> ABSTAIN (out of scope)
```

The router uses `sentence-transformers/all-MiniLM-L6-v2` out of the box — **no local paths or
private packages required**. Two optional environment variables exist for advanced use:
`YAZ_EMBEDDER_PATH` (point at an alternative `Embedder` package; the bundled MiniLM is used if unset)
and `YAZ_TINYSTORIES_DIR` (a TinyStories corpus, only needed for the optional bits-per-character
side-checks).

A trained checkpoint (`checkpoints/yaz_gen_semantic_v2.pt`) ships with the repo; retrain with
`python scripts/train_gen.py configs/semantic_v2.json`.

## Reproduce the results

```bash
python scripts/scaling/s3_route_abstain.py     # abstention risk-coverage (AURC)
python scripts/scaling/s4_create_primitive.py  # the CREATE 4-condition battery
pytest -q                                       # smoke test asserting a headline number
```

All runs are deterministic (seed 2026), CPU. Results write to `results/`.

## What it can do (measured)

| capability | result |
|---|---|
| Edit a fact, no retraining (UPDATE) | in-dist reliability 1.000; edits land 8/8 (first byte) |
| Delete a fact | fact gone, 0 collateral on others |
| Create a new fact | passes the 4-condition battery (monosemantic / local / readable / deletable) |
| Provable per-edit locality | 0/10 collateral, bpc +0.000% across 40 sequential edits |
| No sequential-edit collapse | retention flat 1.000 over 40 edits |
| Paraphrase-robust routing | held-out reach 0.696 (vs 0.216 surface-routing) |
| Abstain when unsure | near-oracle: risk-coverage AURC 0.004 (oracle 0.003) |

## Caveats

- **First-byte editor.** Edits reliably set the answer's **first byte**; multi-byte generation is not
  faithful (full-word transfer ≈ 0.05). Treat the first character as the signal.
- **Routing degrades on hard clues** (≈0.85 on famous entities → ≈0.50 on oblique, name-free ones).
- **Locality is structural** — it holds while no two facts route to the same atom; collisions can occur
  at larger fact counts.
- **Tiny, synthetic scope** — 50 country→capital facts, single seed, CPU. Not validated at scale or on
  open-vocabulary knowledge.
- **Not novel-by-defensibility.** The mechanisms (sentence-embedding-keyed editing, selective
  prediction) are individually present in the published literature. Yaz is a clean, reproducible
  prototype, not a unique capability.

## Layout

```
yaz/            model + semantic router
scripts/        training, eval, and reproduction scripts
data/           synthetic country→capital facts + paraphrase probes
checkpoints/    trained model(s)
results/        result JSON written by the scripts
demo.py         the CLI demo
tests/          smoke test
```

## License

MIT — see [LICENSE](LICENSE).
