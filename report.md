# Citation Reliability of Small Open-Weight LLMs

**Sole author, 2026**

## What I studied

I studied whether two small open-weight language models produce citations you could actually trust. The two models were microsoft/Phi-3-mini-4k-instruct, a 3.8B model from April 2024, and Inpris/humains-junior, also 3.8B. I designed the topics, ran the models, wrote the verifier with AI assistance, and interpreted the results.

My main question was whether citation behavior depends on how well-known a topic is. I split topics into three conditions. Condition A was famous ML topics: Transformer, GANs, ResNet, BERT, GPT-3, ViT, CLIP, Diffusion, AlphaFold, RLHF, Dropout, Batch normalization, Adam, Word2Vec, and Seq2Seq. Condition B was my own lab's topics: lighting-robust semantic mapping, underwater monocular odometry, and 4D Gaussian splatting with rigid-body constraints. Condition C was recent robotics topics: event-based Gaussian splatting SLAM, VLA models, world models, and distributed multi-agent SLAM.

## How I studied it

Each condition had 15 topics. Each topic had 2 prompt templates, and each prompt asked for 5 references. That gives 90 prompts per model and 45 topics in total. I used greedy decoding with max_new_tokens set to 900 and seed 0 on a T4 GPU.

I checked every reference against Crossref (using query.bibliographic), OpenAlex, and arXiv. I matched titles with a similarity threshold of 0.90. Each reference got one label: REAL_OK, REAL_BAD_LINK, WRONG_PAPER_LINK, FABRICATED, or UNPARSABLE. Every model produced 450 references.

For the primary analysis I worked at the topic level. [TODO: define how the topic-level score is computed.] For the p-values below I used [TODO: name the statistical test].

## What I found

At the reference level, Phi-3-mini on condition A had REAL_OK 0.39, REAL_BAD_LINK 0.15, WRONG_PAPER_LINK 0.30, and FABRICATED 0.16. On condition B it had REAL_OK 0.03 and FABRICATED 0.91. On condition C it had REAL_OK 0.00 and FABRICATED 0.94. [TODO: remaining labels for B and C.]

Humains-junior on condition A had REAL_OK 0.07 and FABRICATED 0.66. On condition B it had REAL_OK 0.00 and FABRICATED 0.94. On condition C it had REAL_OK 0.00 and FABRICATED 0.98. [TODO: remaining labels for A, B, and C.]

At the topic level, my primary measure, Phi-3-mini scored A 0.38, B 0.03, and C 0.00. Humains-junior scored A 0.06, B 0.00, and C 0.00.

For Phi-3-mini, the comparisons were A vs B p=8.4e-05, A vs C p=8.5e-06, and B vs C p=0.16. For humains-junior, they were A vs B p=0.018, A vs C p=0.014, and B vs C p=1.0.

The pattern I did not expect was that the failure mode changes with topic familiarity. On famous topics the models often returned real papers but with wrong metadata. On niche and recent topics they returned fully fabricated papers, complete with placeholder authors such as "John Doe" and sequential DOIs. Nothing in the output distinguishes the two cases for a reader.

## What it means

A fluent, well-formatted citation list from a 3.8B model is not evidence that the citations exist. Both models did best on famous topics, and both were weakest on topics from my lab and on recent robotics work, which are exactly the cases where a researcher might lean on a model because direct search is harder. The most damaging part is that fabricated and real-but-wrong citations look the same. A reader skimming authors and titles gets no warning. This means any use of these models for literature search needs an independent check against a real index before the citation is trusted. For my lab topics and for recent robotics, the models were effectively unusable as citation sources.

## Limitations

My study covers two 3.8B models only, so these numbers should not be extended to larger or other models. Greedy decoding with seed 0 gives one deterministic run, so I do not measure variation across sampling settings. [TODO: number of prompt templates was fixed at 2, note whether that limits topic coverage.] The verifier relies on Crossref, OpenAlex, and arXiv, so a real paper that is missing from all three could be scored as fabricated. Title matching at a 0.90 similarity threshold can misjudge near-duplicate titles. [TODO: whether human spot-checks were done and how many.] [TODO: any manual review of UNPARSABLE cases.] The artifact for this work, including data and code, is at https://huggingface.co/buckets/ayansaleem827/fatima-blindspots-eval.
