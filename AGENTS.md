# LFM2-KoEn-Tuning — AGENTS.md

**Generated:** 2026-03-14  
**Project:** Korean-English translation model fine-tuning (LFM2-1.2B)  
**Type:** Public fork (research notebooks)  
**Stack:** Python / Jupyter / Unsloth / GRPO

---

## OVERVIEW

Fine-tuning experiments for Korean-English bidirectional translation using LiquidAI LFM2-1.2B base model. Achieves SOTA performance (CHrF++ 34.61) beating Gemma-3 4B with only 1.2B parameters. Training pipeline includes SFT → GRPO (RL) with COMET + CHrF++ rewards.

**Upstream:** https://github.com/gyunggyung/LFM2-KoEn-Tuning  
**Maintainer:** @gyunggyung (Kiwoong Yeom)

---

## STRUCTURE

```
LFM2-KoEn-Tuning/
  README.md              # Main documentation (Korean)
  README_EN.md           # English documentation
  colab/                 # Google Colab notebooks
    GRPO_v8_adapter_github.ipynb           # RL GRPO (SOTA)
    GRPO_v8_unsloth_vllm_github.ipynb      # RL Unsloth+vLLM
    SFT_colab_github.ipynb                 # SFT Colab style
    SFT_v6.1_curriculum_github.ipynb       # SFT Kaggle style
  kaggle/                # Kaggle notebooks
    SFT_v6.1_curriculum.ipynb
    SFT_v6_200k.ipynb
  evaluation/
    benchmark_flores200.ipynb              # Flores-200 benchmark
  quantization/
    convert_to_gguf_github.ipynb           # GGUF conversion
  dataset/
    samples/                               # Training data samples
    upload_to_hf_github.py                 # HuggingFace upload script
```

**Total:** 29 files, 7 Jupyter notebooks

---

## USAGE

### Quick Start (Colab)

1. Open `colab/SFT_colab_github.ipynb` in Google Colab
2. Select T4 GPU runtime
3. Run all cells to train SFT base model
4. For SOTA: Run `colab/GRPO_v8_adapter_github.ipynb` after SFT

### Inference (GGUF)

```python
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Download GGUF model
model_path = hf_hub_download(
    "gyung/lfm2-1.2b-koen-mt-v8-rl-10k-merged-GGUF",
    "lfm2-1.2b-koen-mt-v8-rl-10k-merged-Q8_0.gguf"
)

# Load model
llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1)

# Translate
prompt = """<|im_start|>system
Translate to Korean.<|im_end|>
<|im_start|>user
Hello, world!<|im_end|>
<|im_start|>assistant
"""
output = llm(prompt, max_tokens=256, stop=["<|im_end|>"], temperature=0.3)
print(output['choices'][0]['text'])
```

---

## BENCHMARK RESULTS

### SOTA Model (v8 RL Adapter)

| Metric | Score | Rank |
|--------|-------|------|
| CHrF++ | 34.61 | #4 (beats Gemma-3 4B) |
| BLEU | 13.21 | - |
| Params | 1.2B | 3x smaller than competitors |

**Training:** 400 steps (0.78 epoch), GRPO with COMET + CHrF++ rewards

### Model Checkpoints

| Model | Description | Link |
|-------|-------------|------|
| v8 Adapter (SOTA) | GRPO RL adapter | [HuggingFace](https://huggingface.co/gyung/lfm2-1.2b-koen-mt-v8-rl-10k-adapter) |
| v8 GGUF | Quantized (Q8_0) | [HuggingFace](https://huggingface.co/gyung/lfm2-1.2b-koen-mt-v8-rl-10k-merged-GGUF) |
| v6.4 Merged | SFT base | [HuggingFace](https://huggingface.co/gyung/lfm2-1.2b-koen-mt-v6.4-merged) |

---

## NOTES

- **Fork status:** Research notebooks, not production code
- **Hardware:** Designed for Google Colab T4 GPU (free tier compatible)
- **License:** Liquid AI LFM Open License v1.0 (commercial use <$10M revenue free)
- **Known issues:** 
  - Proper noun hallucinations (e.g., "George W. Bush" → "조지 워싱턴")
  - Bracket format output (not JSON) — parser compatibility issue
- **Future work:** DPO for hallucination correction (v9)

---

**Last Updated:** 2026-01-03  
**License:** Apache 2.0 (code), Liquid AI LFM Open License v1.0 (model)
