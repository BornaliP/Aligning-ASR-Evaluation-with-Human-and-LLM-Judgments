


# Aligning ASR Evaluation with Human and LLM Judgments

Official repository for the paper:

> **Aligning ASR Evaluation with Human and LLM Judgments: Intelligibility Metrics Using Phonetic, Semantic, and NLI Approaches**  
> Bornali Phukon, Xiuwen Zheng, Mark Hasegawa-Johnson (2025)  
> arXiv: https://arxiv.org/abs/2506.16528

---

## **📌 Overview**

Traditional ASR evaluation metrics such as **Word Error Rate (WER)** and **Character Error Rate (CER)** fail to measure true *intelligibility*, particularly for dysarthric and non‑standard speech. These metrics are sensitive to surface form mismatches but ignore meaning preservation, logical consistency, and recoverability.

This work proposes a unified **intelligibility metric** that integrates:

- **Phonetic similarity** — captures how similar the ASR output sounds to the reference.
- **Semantic similarity** — captures meaning overlap using contextual embeddings.
- **Logical consistency (NLI)** — detects contradictions, hallucinations, and factual inversions.

---

## **🧮 Metric Definition**

The overall intelligibility score is computed as a weighted combination:

```text
M = λ_NLI · NLI + λ_BERT · BERT + λ_PHON · PHON
````

## **Default Weights**

```text
λ_NLI  = 0.4012
λ_BERT = 0.2785
λ_PHON = 0.3201
```

## **🛠 Installation**

Install required dependencies using pip:

```bash
pip install numpy torch transformers bert-score jellyfish editdistance
```

Or using Conda:

```bash
conda create -n semscore python=3.9 -y
conda activate semscore
pip install numpy torch transformers bert-score jellyfish editdistance
```

## **🚀 Usage**

Example usage in Python:

```python
from semscore.metric import SemScore

metric = SemScore()

refs = ["the medicine should not be taken at night"]
hyps = ["the medicine should be taken at night"]

scores = metric.score_all(refs, hyps)
print(scores)
```

Higher values indicate better intelligibility.

## **🔍 Metric Components**

### 1. Phonetic Similarity

* Soundex encoding
* Jaro–Winkler similarity

This captures approximate acoustic similarity.

### 2. Semantic Similarity (BERTScore)

Contextual similarity using pretrained transformer embeddings, robust to paraphrases and rewording.

### 3. Logical Consistency (NLI)

Uses the pretrained RoBERTa NLI model (`ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`).
Inference is performed in both directions (reference → hypothesis and hypothesis → reference), and the entailment probability is used as the NLI score.



