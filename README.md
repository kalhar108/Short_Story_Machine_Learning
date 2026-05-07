# Rewriting Pre-Training Data Boosts LLM Performance in Math and Code

**CMPE 255 — Data Mining | Short Story Assignment**
**Paper:** [arXiv:2505.02881v4](https://arxiv.org/abs/2505.02881)

## Name: Kalhar Mayurbhai Patel
## SJSU ID: 019140511

---

## 📋 Assignment Overview

* 1. Medium Article Link: https://medium.com/@kalharpatel2002/stop-filtering-start-rewriting-how-swallowcode-and-swallowmath-are-quietly-changing-pre-training-9af7d6e6ad83
* 2. Slide Deck (https://docs.google.com/presentation/d/1gT7fIRh8qiDTqAfgocJQ7a2nDc8e3OpF81UuBugR2FQ/edit?usp=drivesdk): *(add after uploading — `swallowcode_short_story.pptx` and `swallowcode_short_story.pdf` are in this repo)*
* 3. YouTube Video Link: *(add after recording — see `video_script.md` for the 19-minute script with per-slide speaker notes)*

This assignment presents a comprehensive analysis of **SwallowCode and SwallowMath** — two openly-licensed pre-training corpora built by Tokyo Tech and AIST that close the "data quality gap" between open and closed-data LLMs. Within a fixed 50-billion-token continual pre-training budget, swapping the code dataset alone boosts Llama-3.1-8B by **+17.0 points on HumanEval** and **+16.1 points on HumanEval+** versus the previous open-source state of the art (Stack-Edu); swapping the math dataset adds **+12.4 on GSM8K** and **+7.6 on MATH**.

The work is highly relevant to CMPE 255 as it addresses core data mining concepts including:

* **Data Quality and Preprocessing** — the central theme of the paper
* **Large-Scale ETL Pipelines** — four-stage transform-and-retain pipeline
* **Feature Engineering for ML** — what counts as a "good" training sample
* **LLM-Driven Data Refinement** — using a 70B model as a programmable data cleaner
* **Ablation Methodology** — controlled experiments, decontamination, generalization checks
* **Cost-Benefit Analysis** — when expensive preprocessing is worth it

---

## 📂 Repository Structure

```
.
├── README.md                          # This file
├── research_pap.pdf                   # Original research paper (Fujii et al., 2026)
├── medium_article.md                  # Full Medium article, ready to publish
├── video_script.md                    # 19-minute speaker script with per-slide notes
├── swallowcode_short_story.pptx       # 18-slide deck (PowerPoint, source format)
├── swallowcode_short_story.pdf        # 18-slide deck (PDF, for upload to SlideShare)
└── figures/                           # Figures cropped from the paper
    ├── figure1_humaneval_comparison.jpg   # Headline comparison chart
    ├── figure2_pipeline.jpg               # Four-stage pipeline diagram
    ├── figure3_filtering.jpg              # Filtering ablation results
    ├── figure4_rewriting.jpg              # Rewriting ablation results
    └── figure5_math.jpg                   # SwallowMath gains
```

---

## 📄 Paper Information

**Title:** Rewriting Pre-Training Data Boosts LLM Performance in Math and Code
**Authors:** Kazuki Fujii, Yukito Tajima, Sakae Mizuki, Masaki Kawamura, Hinari Shimada, Taihei Shiotani, Koshiro Saito, Masanari Oi, Taishi Nakamura, Takumi Okamoto, Shigeki Ishida, Kakeru Hattori, Youmi Ma, Hiroya Takamura, Rio Yokota, Jun Sakuma, Naoaki Okazaki

**Institutions:**
* Institute of Science Tokyo, Department of Computer Science
* National Institute of Advanced Industrial Science and Technology (AIST)
* Institute of Science Tokyo, Institute of Integrated Research, Supercomputing Research Center

**Published:** arXiv:2505.02881, latest revision (v4) March 2026
**Datasets:** [huggingface.co/collections/tokyotech-llm/swallowcode](https://huggingface.co/collections/tokyotech-llm/swallowcode), [huggingface.co/collections/tokyotech-llm/swallowmath](https://huggingface.co/collections/tokyotech-llm/swallowmath)
**Pipeline code:** [github.com/rioyokotalab/swallow-code-math](https://github.com/rioyokotalab/swallow-code-math)

---

## 🎯 Key Contributions of the Paper

### 1. **The transform-and-retain paradigm**

* Don't filter low-quality code out — *rewrite* it. Preserves the diversity of real-world GitHub code while removing noise.
* Stands in contrast to two prevailing approaches: classifier-based filtering (Stack-Edu, FineWeb-Edu) and synthetic data generation from scratch (Cosmopedia, Magpie). Avoids both the "throw away usable signal" problem of the first and the "diversity collapse" problem of the second.

### 2. **A four-stage SwallowCode pipeline**

| Stage | What it does | Sample reduction | Token reduction |
|---|---|---|---|
| Source | the-stack-v2-train-smol-ids (Python) | — | 36.1B → — |
| Stage 1 | Syntax filter via Python `compile()` | 41.0M → 36.7M (−9.7%) | → 30.5B |
| Stage 2 | Pylint filter (≥7.0) + comment-ratio penalty | 36.7M → 24.1M (−34.3%) | → 20.2B |
| Stage 3 | **SGCR** — Style-Guided Code Rewriting (Google Python Style Guide, 10 criteria) | rewritten | reformatted |
| Stage 4 | **SCOR** — Self-Contained Optimization Rewriting (deps, algorithms, instructive examples) | rewritten | **→ 16.1B (final)** |

### 3. **A parallel pipeline for SwallowMath**

* Five-step LLM rewriting prompt for Finemath-4+: strip web boilerplate, remove timestamps, restore missing context, rewrite derivations, format step-by-step solutions.
* Demonstrates the transform-and-retain paradigm is general — not a Python-specific trick.
* Steps 1–2 mirror the syntax/linter filters for code; steps 3–5 mirror SGCR/SCOR.

### 4. **Rigorous ablations and decontamination**

* 13 code ablations, 2 math ablations — each holding everything constant except the target corpus.
* Decontamination against HumanEval, HumanEval+, GSM8K, MATH (exact-match and Jaccard ≥ 0.8): no matches.
* Self-contamination check using GSM-Plus (post-cutoff for the rewriting LLM): SwallowMath still beats Finemath-4+ by 10.77 points (35.75 → 46.52), confirming gains are real.
* Generalization to a different model family: Qwen2-7B + 20B tokens shows the same +10.3 / +10.3 lift over Stack-Edu on HumanEval / HumanEval+.

---

## 🔬 Technical Deep Dive

### Core Concepts

#### The "Data Quality Gap"

Open-weight frontier models (Qwen3, DeepSeek-V3, Llama 3.3) routinely outperform open community models on HumanEval, GSM8K, and MATH — but they refuse to release their pre-training corpora. The open community is left with public datasets like The Stack v1/v2 and Finemath-4+, which rely on rule-based extraction from CommonCrawl or classifier-based filtering. These approaches retain noisy, redundant, or stylistically inconsistent samples. SwallowCode and SwallowMath aim to close this gap with publicly-released, transform-and-retain corpora.

#### Stage 1 — Syntax Error Filtering

Compile every snippet with Python's built-in `compile()`. Drop anything that fails. Reduces the dataset from 41.0M to 36.7M samples.

```python
# Conceptual filter
try:
    compile(code, "<snippet>", "exec")
    keep(code)
except SyntaxError:
    discard(code)
```

#### Stage 2 — Pylint-Based Filtering

Run pylint on each remaining sample. Drop anything with a score below 7.0. The authors disable environment-dependent warnings (e.g., `import-error`) and add a custom comment-ratio penalty that multiplies the pylint score by `(1 − comment_ratio)`, killing files that are mostly comments with minimal real code.

```python
def apply_comment_penalty(score: float, comment_ratio: float) -> float:
    if comment_ratio == 1.0:
        return 0.0
    elif comment_ratio > 0:
        score *= (1 - comment_ratio)
    return score
```

This stage reduces the dataset from 36.7M to 24.1M samples — a 34% drop, but the quality goes up substantially.

#### Stage 3 — SGCR (Style-Guided Code Rewriting)

Llama-3.3-70B-Instruct rewrites every remaining sample to enforce ten criteria from the Google Python Style Guide:

1. Descriptive variable names
2. Useful comments and docstrings
3. Effective type annotations
4. Modular function design
5. Sensible variable scopes / lifetimes
6. Proper error handling
7. Standard indentation and formatting
8. Comments that explain *why*, not *what*
9. Single-responsibility functions and classes
10. Readability-focused formatting

Average input: 836 tokens. Average output: 548 tokens (SGCR generally condenses the data). **Effect: +7 to +9 points on HumanEval.**

#### Stage 4 — SCOR (Self-Contained Optimization Rewriting)

A second pass on top of SGCR. Fixes three failure modes that SGCR alone leaves behind:

* **Missing dependencies** — model imports modules that don't exist or calls undefined helpers.
* **Inefficient algorithms** — naive recursion or O(n²) where O(n) or DP solutions exist.
* **Trivial snippets** — `print("hello")` and similar code that teaches nothing.

Guided by a ten-rule prompt, SCOR makes each snippet self-contained, replaces inefficient algorithms, and turns trivial code into instructive examples. Average input: 548 tokens. Average output: 835 tokens (SCOR generally expands the data). **Effect: +5 to +6 additional points on HumanEval.**

#### Why two passes instead of one?

The authors tried a single combined prompt with all 19 directives. Llama-3.3-70B-Instruct exhibited **instruction drift** — it would do the structural changes but skip the stylistic refinements, or vice versa. Splitting into two focused passes gave more reliable outputs. This is a generalizable lesson about LLM prompting beyond this paper.

#### SwallowMath rewriting prompt (5 steps)

Llama-3.3-70B-Instruct is instructed to:

1. Remove residual web headers, footers, and privacy notices.
2. Eliminate irrelevant metadata (e.g., question/answer timestamps).
3. Restore missing context in incomplete questions or answers.
4. Rewrite derivation steps to be concise yet comprehensive.
5. Provide clear step-by-step solutions.

Steps 1–2 are analogous to syntax/linter filtering for code; steps 3–5 are analogous to SGCR/SCOR.

---

### Experimental Results

#### Code: SwallowCode vs. existing public corpora

All experiments use Llama-3.1-8B continually pre-trained for 50 billion tokens, with the only variable being the Python code subset. Final scores at 50B tokens:

| Corpus | HumanEval pass@1 | HumanEval+ pass@1 |
|---|---|---|
| The Stack v1 (Python) | 35.6% | 31.8% |
| CodeParrot-Clean | 36.2% | 31.3% |
| The Stack v2-Smol | 37.0% | 31.4% |
| Stack-Edu | 37.0% | 32.0% |
| **SwallowCode (ours)** | **54.0%** ( **+17.0** ) | **48.1%** ( **+16.1** ) |

#### Math: SwallowMath vs. Finemath-4+

| Corpus | GSM8K | MATH |
|---|---|---|
| Finemath-4+ | 52.9% | 24.0% |
| **SwallowMath (ours)** | **65.4%** ( **+12.4** ) | **31.6%** ( **+7.6** ) |

#### Ablation: where the SwallowCode gains come from

| Stage | HumanEval pass@1 (50B tokens) | Cumulative gain |
|---|---|---|
| The Stack v2 (raw) | 37.0% | baseline |
| + Syntax filter (Stage 1) | 37.9% | +0.9 |
| + Pylint filter (Stage 2) | 39.5% | +2.5 |
| + SGCR (Stage 3) | 48.6% | +11.6 |
| + SCOR (Stage 4) | 54.0% | **+17.0** |

**The two filtering stages contribute ~2.5 points combined. The two rewriting stages contribute ~14.5 points combined.** Rewriting is roughly seven times more impactful than filtering.

#### Generalization to Qwen2-7B (Appendix B)

20-billion-token continual pre-training from Qwen2-7B, with a 70% English / 20% code / 10% math mixture:

| Corpus | HumanEval | HumanEval+ |
|---|---|---|
| Stack-Edu (Qwen2-7B) | 39.3 | 34.3 |
| **SwallowCode (Qwen2-7B)** | **49.6** ( **+10.3** ) | **44.6** ( **+10.3** ) |

The dataset is genuinely better, not Llama-tuned.

#### Self-contamination check (GSM-Plus, post-cutoff for rewriter)

| Corpus | GSM-Plus accuracy |
|---|---|
| Finemath-4+ | 35.75 |
| **SwallowMath** | **46.52** |

The rewriter (Llama-3.3-70B-Instruct) cannot have memorized GSM-Plus during training. Gains hold, confirming they are real.

#### Cost-benefit summary

| Item | Cost |
|---|---|
| SwallowCode rewriting (full 16.1B tokens) | ~23,700 H100 GPU-hours |
| SwallowMath rewriting (~2.3B tokens) | ~3,400 H100 GPU-hours |
| Per-experiment continual pre-training (50B tokens) | ~1,580 H100 GPU-hours |
| Net rewriting overhead vs. LLM-scoring | ~1.22× compute |
| Net data efficiency | rewritten 50B ≈ unrefined ≫100B |

---

## 🔑 Key Takeaways

### Why This Paper Matters for Data Mining

1. **Data quality > model quality at this stage.** A 50B-token continual pre-train on better data beats a 50B-token continual pre-train on more data. Training data is the most underleveraged lever in LLM development.
2. **LLM-as-data-cleaner is a new design pattern.** Beyond classifiers, beyond synthetic generation. The 70B model becomes a programmable ETL transform.
3. **Decoupled prompting beats monolithic prompting.** Two prompts with focused objectives beat one prompt with 19 directives. Generalizes far beyond this paper.
4. **Ablations are how you earn trust.** 13 controlled code experiments + decontamination + cross-model generalization is what separates a credible paper from a hype paper.
5. **Open science compounds.** A 23,700 H100-hour rewriting cost paid once becomes a permanent community asset. That economic model deserves to be the default for open data work.

### Relevance to CMPE 255 Topics

This work connects to multiple course themes:

* **Data Quality & Preprocessing:** the central problem framing of the paper
* **Feature Engineering:** rewriting code is essentially feature engineering at the token level
* **Large-Scale ETL:** a four-stage pipeline operating on 41M samples / 36B tokens
* **Ablation Methodology:** controlled isolation of design choices
* **Statistical Validity:** decontamination, cross-model generalization, post-cutoff probes
* **Cost-Benefit Analysis:** when does expensive preprocessing pay off?
* **Bias and Fairness:** rewriter biases, source-data biases, benchmark biases
* **Open Science / Reproducibility:** datasets, prompts, checkpoints all released

### Real-World Impact

**Open-source LLM development:**
* SwallowCode/SwallowMath drop into any continual pre-training run as a near-free upgrade.
* The pipeline code is reusable for new languages, new domains, or new rewriter LLMs.

**Educational content generation:**
* The same SCOR-style rewriting could turn noisy textbook PDFs or lecture transcripts into clean, self-contained study material.

**Scientific reproducibility:**
* Filling missing context in incomplete benchmarks or stripping web boilerplate from harvested papers — the SwallowMath playbook generalizes.

**Code quality automation:**
* SGCR is essentially a Style-Guide-as-a-Service. The same prompt could be applied to legacy codebases as a refactoring assistant.

### Future Research Directions

1. **Multi-language SwallowCode:** the pipeline is in principle language-agnostic; empirical validation on JavaScript, Rust, C++ is the obvious next step.
2. **Cascade with newer rewriters:** as Llama 4, Qwen3, gpt-oss arrive, regenerate SwallowCode and measure the lift.
3. **SCOR without SGCR:** the authors deferred this ablation; it might reveal whether semantic rewriting alone is sufficient.
4. **Beyond pre-training:** apply transform-and-retain to fine-tuning, RLHF, and continual learning corpora.
5. **Theoretical framing:** when *should* rewriting beat filtering? Is there a dataset-quality regime where filtering is enough?
6. **Bias auditing:** systematic study of stylistic biases the rewriter introduces.

---

## 📚 References

### Primary Source

Fujii, K., Tajima, Y., Mizuki, S., Kawamura, M., Shimada, H., Shiotani, T., Saito, K., Oi, M., Nakamura, T., Okamoto, T., Ishida, S., Hattori, K., Ma, Y., Takamura, H., Yokota, R., Sakuma, J., & Okazaki, N. (2026). **Rewriting Pre-Training Data Boosts LLM Performance in Math and Code.** arXiv:2505.02881v4. https://arxiv.org/abs/2505.02881

### Related Work Cited

**Open code corpora:**
* Kocetkov et al. (2022) — *The Stack: 3 TB of Permissively Licensed Source Code*
* Lozhkov et al. (2024) — *StarCoder 2 and The Stack v2*
* Allal et al. (2025) — *SmolLM2 / Stack-Edu*

**Open math corpora:**
* Paster et al. (2023) — *OpenWebMath*
* Han et al. (2024) — *InfiMM-WebMath-40B*
* Zhou et al. (2025) — *MegaMath*

**LLM-driven data refinement:**
* Ben Allal et al. (2024) — *Cosmopedia* (synthetic textbooks)
* Maini et al. (2024) — *Rephrasing the Web*
* Su et al. (2025) — *Nemotron-CC*
* Jain et al. (2023) — *LLM-Assisted Code Cleaning for Training Accurate Code Generators* (closest prior work, instruction-tuning scope)
* Penedo et al. (2024) — *FineWeb / FineWeb-Edu*
* Xu et al. (2024) — *Magpie*

**Base models referenced:**
* Grattafiori et al. (2024) — *The Llama 3 Herd of Models*
* Yang et al. (2024) — *Qwen2*
* Yang et al. (2025) — *Qwen3*
* DeepSeek-AI et al. (2025) — *DeepSeek-V3*
* Abdin et al. (2024) — *Phi-4*
* OLMo Team (2025) — *2 OLMo 2 Furious*
* NVIDIA et al. (2025) — *Nemotron-H*

**Evaluation benchmarks:**
* Chen et al. (2021) — *HumanEval (Evaluating Large Language Models Trained on Code)*
* Liu et al. (2023) — *HumanEval+ (Is your code generated by ChatGPT really correct?)*
* Cobbe et al. (2021) — *GSM8K*
* Hendrycks et al. (2021) — *MATH*, *MMLU*
* Li et al. (2024) — *GSM-Plus*
* Austin et al. (2021) — *MBPP*

**Training infrastructure:**
* Shoeybi et al. (2020) — *Megatron-LM*
* Narayanan et al. (2021) — *Efficient Large-Scale LM Training on GPU Clusters*
* Gao et al. (2024) — *lm-evaluation-harness*

---

## 🛠️ Tools and Technologies

**Pipeline:**
* Python `compile()` for syntax filtering
* `pylint` for linter-based filtering
* Custom comment-ratio scoring heuristic
* Llama-3.3-70B-Instruct as the rewriting LLM
* vLLM 0.7.2 + PyTorch 2.5.1 for inference

**Continual pre-training:**
* Megatron-LM (mcore 0.9.0)
* PyTorch 2.5.0, TransformerEngine 1.12, CUDA 12.4
* H100 (94 GB) supercomputer (TSUBAME 4.0, Institute of Science Tokyo)
* DP=32, TP=2, sequence parallelism, distributed optimizer

**Evaluation:**
* `evalplus` (HumanEval / HumanEval+)
* `lm-evaluation-harness` (GSM8K, MATH, MMLU, BBH, OpenBookQA, TriviaQA, HellaSwag, SQuAD 2.0, XWinograd)
* GSM-Plus for post-cutoff contamination probe

---

## 📊 Additional Insights

### What changes in token counts during rewriting?

| Stage | Avg input tokens | Avg output tokens |
|---|---|---|
| SGCR | 836 | 548 |
| SCOR | 548 | 835 |

SGCR generally compresses (removes verbose comments, dead code). SCOR generally expands (inlines dependencies, adds optimized algorithms, instructive examples).

### Syntax error rates after each rewriting stage

Measured on a random 100K-sample subset:
* Post-SGCR: 0.73% syntax errors
* Post-SCOR: 0.46% syntax errors

Errors do not accumulate across the two-stage pipeline — they actually *decrease*, suggesting SCOR cleans up some of SGCR's slips.

### Why MBPP was excluded from evaluation

MBPP's evaluation harness expects functions named like `is_Power_Of_Two` (camelCase mixed with capitalized words, non-PEP-8). SGCR enforces snake_case per the Google Python Style Guide. Models trained on SGCR data correctly write `is_power_of_two`, but MBPP marks the answer as "is not defined." The authors disclose this transparently and exclude MBPP — a benchmark artifact, not a capability regression. **General lesson: benchmarks embed assumptions; always interrogate what they're actually measuring.**

### English-heavy mixture sanity check (Appendix A.5)

To rule out the possibility that the Japanese-heavy training mixture interacts with the English-centric benchmarks in some non-obvious way, the authors re-ran the experiment with a 70% English / 20% code / 10% math mixture. SwallowCode/SwallowMath gains hold:

| Benchmark | Score |
|---|---|
| GSM8K | 0.700 |
| MATH | 0.354 |
| HumanEval | 0.583 |
| HumanEval+ | 0.540 |

Higher absolute scores than the Japanese-heavy mixture (Llama-3.1-8B is English-trained), but the *relative improvement of SwallowCode/SwallowMath over baselines* remains robust.

---

## **Original Paper:** [https://arxiv.org/abs/2505.02881](https://arxiv.org/abs/2505.02881)
## **Pipeline Repository:** [https://github.com/rioyokotalab/swallow-code-math](https://github.com/rioyokotalab/swallow-code-math)
## **Datasets:** [https://huggingface.co/collections/tokyotech-llm/swallowcode](https://huggingface.co/collections/tokyotech-llm/swallowcode)

---

## 📄 License and Attribution

This assignment is submitted for **CMPE 255 — Data Mining**. All analysis, written commentary, the slide deck, and the Medium article are original work by the author. All figures used are from the original paper, used under fair-use for educational review purposes. The paper being analyzed is:

```
@article{fujii2026rewriting,
  title   = {Rewriting Pre-Training Data Boosts LLM Performance in Math and Code},
  author  = {Fujii, Kazuki and Tajima, Yukito and Mizuki, Sakae and Kawamura, Masaki
             and Shimada, Hinari and Shiotani, Taihei and Saito, Koshiro and Oi, Masanari
             and Nakamura, Taishi and Okamoto, Takumi and Ishida, Shigeki and Hattori, Kakeru
             and Ma, Youmi and Takamura, Hiroya and Yokota, Rio and Sakuma, Jun
             and Okazaki, Naoaki},
  journal = {arXiv preprint arXiv:2505.02881},
  year    = {2026},
  note    = {v4, March 2026},
  url     = {https://arxiv.org/abs/2505.02881}
}
```

The SwallowCode and SwallowMath datasets are released under the Llama 3.3 Community License.

---

**Last Updated:** Spring 2026
**Course:** CMPE 255 — Data Mining
**Assignment:** Individual Paper Analysis and Presentation (Short Story)
