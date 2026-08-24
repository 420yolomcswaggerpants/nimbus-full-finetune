---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B
tags:
- fine-tuned
- qwen
- nimbus-coffee
- customer-support
- full-fine-tuning
metrics:
- loss
---

# Nimbus Coffee Assistant - Full Fine-Tuned

A full fine-tuned Qwen 2.5 0.5B model trained on 80 Q&A pairs for Nimbus Coffee customer support. Trained as a comparison against LoRA fine-tuning.

## Model Description

This model is a full fine-tune of Qwen 2.5 0.5B—all weights were updated during training, not just LoRA adapters. It's trained to answer customer questions about Nimbus Coffee, a fictional Portland-based coffee roaster.

## Training Details

- **Base model:** Qwen/Qwen2.5-0.5B
- **Fine-tuning type:** Full (all weights updated)
- **Dataset:** 80 Q&A pairs
- **Epochs:** 5 (reduced from 25 to prevent overfitting)
- **Learning rate:** 5e-5 with linear decay
- **Weight decay:** 0.01
- **Batch size:** 1 with gradient accumulation (8 steps)
- **Training time:** 14 minutes 25 seconds
- **Hardware:** CPU

## Loss Progression (5 epochs)

| Epoch | Loss |
|-------|------|
| 1 | 1.726 |
| 2 | 0.570 |
| 3 | 0.288 |
| 4 | 0.221 |
| 5 | 0.193 |

## Comparison with LoRA Version

### Training Metrics

| Method | Final Loss | Training Time | Epochs |
|--------|-----------|---------------|--------|
| LoRA 0.5B | 0.10 | ~2 hours | 25 |
| Full FT 0.5B (25 epochs) | 0.158 | 1h 12min | 25 |
| Full FT 0.5B (5 epochs) | 0.193 | 14min 25s | 5 |

### Evaluation on Test Questions

Tested on 10 questions (4 training + 6 unseen).

| Question Type | LoRA | Full FT (5 epochs) |
|--------------|------|-------------------|
| Training questions | 3/4 correct | 1/4 correct |
| Unseen questions | 0/6 correct | 1/6 partially correct |

### Key Findings

1. **LoRA outperformed full fine-tuning on this dataset.** LoRA answered more training questions correctly and hallucinated less on unseen questions.

2. **Full fine-tuning overfit despite loss improvements.** The 25-epoch version achieved 0.158 loss but performed worse than the 5-epoch version on generalization.

3. **80 examples is too small for full fine-tuning.** Full fine-tuning updates all weights and needs larger datasets to learn generalizable patterns without memorizing.

4. **LoRA is better for small-data scenarios.** The adapter approach constrains learning, which prevents overfitting when data is limited.

5. **Reducing epochs helped but didn't solve the problem.** The 5-epoch version generalized slightly better than 25 epochs but still hallucinated on unseen questions.

## Usage

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "420yolomcswaggerpants/nimbus-full-finetune-0.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "### Instruction:\nWhat coffees do you sell?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

## Limitations

- Trained on only 80 examples—limited domain coverage
- Overfits to training phrasing
- Hallucinates on out-of-domain questions
- CPU training limited batch size and gradient accumulation
- Generation can produce repetition loops (seen with "no hassle" repeating)
- Full fine-tuning underperformed LoRA on this dataset size

## Intended Use

Demonstrates full fine-tuning as an alternative to LoRA. Key lesson: full fine-tuning requires larger datasets than LoRA to be effective. This model serves as a comparison point for the LoRA version.

## Conclusion

Full fine-tuning is not superior to LoRA on small datasets. For 80 examples, LoRA achieved better generalization. Full fine-tuning would likely outperform LoRA with 500+ examples, but that requires significantly more training data.