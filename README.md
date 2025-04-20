# Experiment Results: Full vs LoRA (RoBERTa Base)

## Accuracy

| Dataset   | Full  | LoRA  | Δ Accuracy |
|-----------|-------|-------|------------|
| MNLI      | 0.8611 | 0.8311 | -0.0300    |
| MRPC      | 0.8725 | 0.6838 | -0.1887    |
| QNLI      | 0.9055 | 0.8810 | -0.0245    |
| QQP       | 0.8883 | 0.8561 | -0.0322    |
| RTE       | 0.5271 | 0.5271 | +0.0000    |
| SST-2     | 0.9232 | 0.9278 | **+0.0046** |

---

## GPU Memory Usage (Batch size: 16, Optimizer: Adam)

| Dataset   | Full (MB) | LoRA (MB) | Memory Savings (%) |
|-----------|-----------|-----------|--------------------|
| MNLI      | 6387      | 4775      | **25.27%**         |
| MRPC      | 6465      | 4813      | **25.54%**         |
| QNLI      | 6010      | 4334      | **27.88%**         |
| QQP       | 6063      | 4431      | **26.91%**         |
| RTE       | 6101      | 4439      | **27.22%**         |
| SST-2     | 6465      | 4827      | **25.35%**         |

---

## Training Speed Improvement (Requests per second)

| Dataset   | Full (rps) | LoRA (rps) | Speedup (×) |
|-----------|------------|------------|-------------|
| MNLI      | 5.677      | 7.620      | **1.34×**    |
| MRPC      | 5.181      | 7.149      | **1.38×**    |
| QNLI      | 5.831      | 7.523      | **1.29×**    |
| QQP       | 5.635      | 7.430      | **1.32×**    |
| RTE       | 4.867      | 6.647      | **1.37×**    |
| SST-2     | 5.740      | 7.869      | **1.37×**    |

---

## Conclusions

- **Accuracy**: LoRA results in a slight decrease in accuracy for most datasets, with the most significant drop observed in MRPC (-0.1887). However, the performance remains competitive across other datasets, with a small gain in SST-2 (+0.0046).

- **Memory Usage**: LoRA provides substantial memory savings by reducing memory consumption by approximately 25-28%. This is because LoRA does not store full-sized matrices for gradients, as in the Adam optimizer, but instead stores a smaller low-rank matrix. This reduction is achieved without sacrificing much performance, making it highly memory-efficient.

- **Training Speed**: LoRA speeds up training by approximately 1.3x to 1.4x across all datasets, as shown by the increase in requests per second (rps). This indicates faster training compared to the full model, which can be particularly useful for large-scale model training where time and resources are critical.

In conclusion, LoRA provides a tradeoff: while there is a slight accuracy drop, it offers significant memory savings and faster training, making it an attractive option for scenarios with limited resources or where faster training is required. This study is based on RoBERTa Base, highlighting the effectiveness of LoRA in large transformer models.
