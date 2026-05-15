# 🏥 Medical LLM Fine-tuning using Mistral-7B + QLoRA
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Tracking-orange)

## 🎯 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/mou11/medical-llm-finetune)

## 🔍 Overview
This project fine-tunes Mistral-7B-Instruct-v0.2 on USMLE-style medical question answering data using QLoRA (Quantized Low-Rank Adaptation). By freezing the base model in 4-bit precision and training only small LoRA adapter layers (0.19% of parameters), a 7 billion parameter LLM is fine-tuned on a single free T4 GPU in under 3 hours.

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Loss | 1.039 |
| Validation Loss | 0.987 |
| USMLE Sample Accuracy | 30% (vs 20% random baseline) |
| Trainable Parameters | 13.6M (0.19% of 7.2B) |
| Training Time | ~3 hours on T4 GPU |

## 🤗 Model
👉 [Model on Hugging Face](https://huggingface.co/mou11/medical-mistral-7b-qlora)

## 📈 Experiment Tracking
👉 [W&B Training Logs](https://wandb.ai/moumitaroy4455-stu/medical-llm-finetune)

## 🏗️ Architecture
- **Base Model:** mistralai/Mistral-7B-Instruct-v0.2
- **Fine-tuning Method:** QLoRA (4-bit NF4 quantization + LoRA adapters)
- **LoRA Config:** rank=16, alpha=32, target modules: q_proj, k_proj, v_proj, o_proj
- **Dataset:** medalpaca/medical_meadow_medqa — 9,160 training samples
- **Evaluation:** USMLE-style multiple choice medical questions
- **RAG Pipeline:** Hybrid search combining dense vector retrieval + sparse BM25
- **Inference:** INT8 quantization for deployment efficiency

## 📁 Dataset
[medical_meadow_medqa](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa) from HuggingFace
- Total: 10,178 USMLE-style clinical vignette questions
- Train: 9,160 samples
- Test: 1,018 samples

## 🔑 Why QLoRA?
Full fine-tuning of a 7B model requires ~112GB of GPU memory — impossible on free hardware. QLoRA reduces this to ~16GB by:
- Quantizing the base model to 4-bit NF4 precision (frozen)
- Training only small rank-16 adapter matrices injected into attention layers
- Using paged_adamw_8bit optimizer to reduce memory spikes

## 🛠️ Tech Stack
- PyTorch
- HuggingFace Transformers + TRL + PEFT
- Mistral-7B-Instruct-v0.2
- Weights & Biases
- Kaggle T4 GPU (free tier)

## 🚀 How to Run
1. Open `medical_llm_finetune.py` in Kaggle
2. Connect to T4 x2 GPU runtime
3. Add your WANDB_API_KEY and HF_TOKEN to Kaggle Secrets
4. Run all cells in order

## 📌 Project Status
✅ Dataset loaded and formatted
✅ Mistral-7B loaded in 4-bit QLoRA
✅ Model fine-tuned on USMLE medical QA
✅ Experiment tracked with Weights & Biases
✅ Model adapter pushed to Hugging Face
✅ Gradio demo deployed on Hugging Face Spaces
