# Citation reliability of two small open-weight LLMs

## What this is

I use small LLMs to sketch citation lists while writing papers on robot perception and Gaussian-splatting localisation. I know they hallucinate references. I wanted to know whether the hallucination rate changes with topic popularity, and whether the model shows any sign of uncertainty when it is guessing.

This bucket contains the full evaluation: topics, prompts, raw model outputs, verification results, and the statistics.

## Models evaluated

- `microsoft/Phi-3-mini-4k-instruct` (3.8B, April 2024)
- `Inpris/humains-junior` (3.8B)

Both were run with greedy decoding (`do_sample=False`), `max_new_tokens=900`, seed 0. This is the same decoding configuration in both notebooks, so the comparison is like-for-like on the generation side.

I picked Phi-3-mini because it is small enough to fit on a free Colab T4 and its model card claims strong factual grounding. I added humains-junior as a second candidate after I saw that Phi-3-mini was failing even my control topics.

## Question 1 — the blind spot

Every time I write a related-work section, I use a small LLM to sketch out the citation list. I know it hallucinates. I check the references. That is not the blind spot — that is the known risk.

The blind spot is that the hallucinations are not evenly distributed, and the model's confidence is not calibrated to that distribution.

I noticed it while writing a paper on Gaussian-splatting-based localisation. I asked a small model for five references on 4D Gaussian splatting. It returned a paper by "John Doe, Jane Smith" at a fake ICRA DOI. I deleted that one, asked again, got the same list with a different fake DOI. I asked for a well-known topic — Transformer — and got *Attention Is All You Need*, but with a fabricated author list and a DOI that resolved to a Canadian journal on neural networks that does not exist. The model was confident in both cases. The formatting was identical. There was no signal in the output itself that one was real and one was not.

The pattern that actually harms a researcher is this: for famous topics, the model gets the *titles* mostly right but the *metadata* mostly wrong. For niche topics, it gets everything wrong. So the reference list looks half-credible. You fix the two obvious fakes, trust the other three, and one of the three has a wrong DOI that Crossref will not resolve, and you only discover it at submission. Or worse, you never discover it, and the paper goes out with a dead citation.

The standard benchmarks do not catch this because they test either end of the distribution. Retrieval benchmarks test whether the model can copy a passage. Citation benchmarks test famous papers, or test whether a model refuses to answer. Nobody measures the calibration: how does the *rate of subtle metadata errors* vary with topic popularity, and does the model flag any uncertainty at all?

This is underdiscussed for a specific reason. Researchers in NLP and vision work in the fields that are heaviest in the training data. Their "control" condition (Transformer, BERT, ResNet) mostly passes, so their trust in the tool is roughly correct. Researchers in smaller communities — robotics, remote sensing, and anything outside English-language venues — get the failure mode *without* the familiar calibration. A robotics PhD in Lahore and a NLP PhD in Berkeley use the same model and get the same-looking output, but the robotics PhD's references are three times more likely to be silently wrong. That asymmetry is not in any benchmark I could find.

## What I did

Three conditions, 15 topics each, two prompts per topic, 5 references requested per prompt:

- **A** — famous ML topics: Transformer, GANs, ResNet, BERT, GPT-3, ViT, CLIP, diffusion models, AlphaFold, RLHF, Dropout, Batch normalization, Adam, Word2Vec, Seq2Seq.
- **B** — topics from my own lab work: lighting-robust semantic mapping, underwater monocular odometry, 4D Gaussian splatting with rigid-body constraints.
- **C** — recent robotics topics that postdate the models' training data: event-based Gaussian splatting SLAM, VLA models, world models, distributed multi-agent SLAM.

Every reference the model produced was verified against Crossref and OpenAlex, and its own DOI/arXiv link was resolved to check what it actually points to. Labels:

- `REAL_OK` — paper exists, link resolves to it
- `REAL_BAD_LINK` — paper exists, link is dead
- `WRONG_PAPER_LINK` — link is valid but points at a different paper
- `FABRICATED` — no matching paper in either index
- `UNPARSABLE` — no valid JSON in the response

## Results

**Reference-level (Wilson 95% CI in brackets):**

| Model | Cond | n | REAL_OK | REAL_BAD_LINK | WRONG_PAPER_LINK | FABRICATED |
|---|---|---|---|---|---|---|
| Phi-3-mini | A | 145 | 0.39 [0.31, 0.47] | 0.15 [0.10, 0.22] | 0.30 [0.23, 0.38] | 0.16 [0.11, 0.23] |
| Phi-3-mini | B | 150 | 0.03 [0.01, 0.07] | 0.00 | 0.07 [0.04, 0.12] | 0.91 [0.85, 0.94] |
| Phi-3-mini | C | 140 | 0.00 [0.00, 0.03] | 0.00 | 0.06 [0.03, 0.12] | 0.94 [0.88, 0.97] |
| humains-junior | A | 135 | 0.07 [0.04, 0.12] | 0.08 [0.05, 0.14] | 0.19 [0.13, 0.27] | 0.66 [0.58, 0.73] |
| humains-junior | B | 140 | 0.00 | 0.00 | 0.06 [0.03, 0.12] | 0.94 [0.88, 0.97] |
| humains-junior | C | 140 | 0.00 | 0.00 | 0.02 [0.01, 0.06] | 0.98 [0.94, 0.99] |

**Topic-level (primary result, one number per topic):**

| Model | A | B | C |
|---|---|---|---|
| Phi-3-mini | 0.38 (CI 0.23–0.54) | 0.03 (0.00–0.07) | 0.00 |
| humains-junior | 0.06 (CI 0.01–0.12) | 0.00 | 0.00 |

Mann-Whitney U, topic level:

- Phi-3: A vs B p = 8.4e-05; A vs C p = 8.5e-06; B vs C p = 0.16
- humains-junior: A vs B p = 0.018; A vs C p = 0.014; B vs C p = 1.0

The reference-level Fisher test is also reported in the notebook, but I do not treat it as primary. References from one topic are not independent, so its p-values are too small.

## What I found

Three things, in order of how much they surprised me.

**1. Both models fail my control condition.** Phi-3-mini's `REAL_OK` rate on famous topics is 0.39. That is the *best* case. On Transformer, it returned *Attention Is All You Need* with a fabricated author list (extra names like "Jakob Uszavecz, Adrian Bansal, Esme Taylor, Rishad Zamir"), and a second paper that does not exist. On ViT, it invented a paper called "ViT: All You Need for State-of-the-Art Image Classification" with a DOI of the form `10.48550/arXiv.2002.12092`, which belongs to a different arXiv paper. On CLIP, all five entries were fake. Even the "real" entries often carry wrong authors, wrong venues, or wrong DOIs.

**2. The gradient exists, but only between A and B.** A vs B is significant for both models. A vs C is significant for both models. But B vs C is flat (Phi-3 p = 0.16, humains-junior p = 1.0). This is a floor effect, not evidence that recency does not matter. Once the model is below a certain familiarity threshold, both "niche but pre-cutoff" and "recent" topics collapse to near-zero `REAL_OK`. I cannot distinguish "the model has never seen this" from "the model has seen it but badly."

**3. The failure mode changes across conditions.** Condition A produces real titles with wrong metadata. Condition B and C produce fully fabricated titles, often by the same mechanism: take the topic phrase, prepend a plausible adjective, add "John Doe, Jane Smith", and increment a fake DOI. The humains-junior output on "4D Gaussian splatting for robot localisation" contains DOIs of the form `10.1109/ICRA.2022.1000100` through `1000500`. These are sequential. They are clearly generated, not retrieved.

That sequential-DOI pattern is the thing I would most want the next benchmark to look at. It is trivial to detect, and its presence is a strong signal that the model is in a different regime from "recalling a real paper with a wrong year."

## Question 3 — path forward

The problem is not that the model does not know facts. It is that the decoder is free to emit a DOI string that was never in its training data. Fixing that requires changing what the model is allowed to output, not just what it knows.

Three layers, in increasing order of effort:

**Retrieval with a locked vocabulary.** Before generating, query OpenAlex or Crossref with the topic, retrieve the top 20 candidates, and put them in the context. Then constrain the generation so the `doi` field must be one of the DOIs that appear in the retrieved set. If the model wants to cite a paper that is not in the retrieved set, it must say so in a separate `uncertain` field rather than emit a DOI. This alone would eliminate most fabrication because the model literally cannot invent a DOI that is not in its allowed output set. The trade-off is that it cannot cite papers the retriever missed, so the retriever matters more than the model.

**A curation strategy that matches the failure mode.** OpenAlex has a dump of ~250M works with verified DOIs. Filter to works with (a) at least one arXiv ID and (b) at least five citations. Build training pairs of the form *topic phrase → list of (title, DOI, authors, year)*. Emphasise non-English venues and non-Western conferences, which the current training data underrepresents. The point of the filter is that every training target is a DOI that resolves today, so the model is trained to recall real DOIs, not plausible-looking ones.

**A dual-head decoder.** This is speculative. Keep the language head, add a second head with a fixed vocabulary of, say, 10 million known DOIs. When the model emits a citation, the DOI must come from the second head. The two heads share the same hidden state, so the model still knows *which* paper it wants — the constraint is just on *how* it is written out. This is essentially a sparse softmax over a frozen vocabulary, and it could be trained with a contrastive loss that pulls the hidden state of "Attention Is All You Need" toward its real DOI and pushes it away from every other DOI in the batch.

If I had one weekend and no compute, I would do the first option. If I had a research budget, I would do all three in that order and publish the calibration curve from the first as the baseline for the other two. The measure that matters is not "does the model cite real papers on famous topics" — it is "does the FABRICATED rate stay flat as you move from famous to niche to recent." Neither model I tested holds that flat.

## Limitations

- **Audit pending.** I ran the verification automatically with `SIM_THRESHOLD = 0.90` on title similarity. Crossref and OpenAlex do not index every paper. A sample of 30 `FABRICATED` rows was drawn for manual checking; the false-alarm rate is reported in `audit_sample.csv` and summarised here as [X/30 = Y, 95% CI ...]. [Update this number once you fill the audit.]
- **Floor effect in B and C.** Both are near zero for both models. This is why I cannot say whether recency matters separately from niche-ness.
- **Titles with wrong authors may still be labelled `REAL_OK`.** The verifier does not check authors or years. The A results are therefore an upper bound on correctness; the real error rate is higher.
- **The similarity threshold is a design choice.** 0.90 was picked before the runs. A different value would shift some boundary cases between `REAL_OK` and `FABRICATED`.
- **One template pair.** Two phrasings. A different phrasing might produce different rates. The two are averaged in the topic-level analysis.
- **humains-junior output was written to a Windows path.** The `output.jsonl` for that model is not included in this bucket. Its `verified.csv` and label counts are.

## Files

- `prompts.jsonl` — 90 prompts (15 topics × 2 templates × 3 conditions)
- `outputs.jsonl` — raw Phi-3-mini responses
- `verified.csv` — one row per reference, with label and match info
- `Inpris-humains-junior.ipynb`, `microsoftPhi-3-mini-4k-instruct.ipynb` — the run notebooks

