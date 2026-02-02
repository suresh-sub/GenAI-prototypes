# LLM Fine-tuning: Sentiment Analysis with LoRA

This repository demonstrates **parameter-efficient fine-tuning (PEFT)** of small causal language models **DistilGPT-2** and **Qwen2.5-0.5B** for sentiment analysis. 

The project highlights how to adapt Large Language Models for specific classification tasks using **LoRA (Low-Rank Adaptation)**, enabling high-performance training on mid-range consumer hardware.

---

## 🎯 Problem Setup

Unlike traditional sentiment analysis that uses a classifier head, this project frames the task as **instruction-style text generation**. The model is prompted to generate a specific text label ("positive" or "negative") based on a movie review.

- **Dataset:** Rotten Tomatoes (Binary Sentiment Classification)
- **Framing:** Causal Language Modeling (Text-to-Text)
- **Hardware:** NVIDIA RTX 3070 (8GB VRAM)

## 🛠️ Technical Stack

- **Frameworks:** PyTorch, Hugging Face `transformers`
- **PEFT Library:** `peft` for implementing LoRA
- **Models:**
  - `distilgpt2` (82M parameters): A lightweight transformer for speed.
  - `Qwen2.5-0.5B` (494M parameters): A modern, dense small language model.

## 🏗️ Methodology

### LoRA Configuration
To maximize efficiency, the implementation targets the query and value projections within the transformer blocks using the following hyperparameters:
- **Rank ($r$):** 8
- **Alpha:** 32
- **Dropout:** 0.1
- **Target Modules:** `q_proj`, `v_proj`

### Training Process
By freezing the base model weights and training only the injected low-rank matrices, the trainable parameter count was reduced to **less than 1%** of the total model size. This allowed for significant memory savings and faster iteration cycles.

## 📊 Key Results & Insights

- **Efficiency:** Successfully fine-tuned the 0.5B parameter Qwen model on an 8GB GPU.
- **Performance:** `Qwen2.5-0.5B` showed superior semantic understanding and higher accuracy compared to `distilgpt2`, justifying the increased parameter count.
- **Inference:** The resulting LoRA adapters are lightweight (~10MB), making them easily portable for edge deployment.

## 🚀 Future Directions

- **Quantization:** Combining LoRA with 4-bit quantization (QLoRA) to fine-tune even larger models.
- **Multi-Domain:** Evaluating the instruction-style prompt on financial and technical datasets.
- **Model Compression:** Exploring adapter merging for zero-latency inference.

## ⚙️ Installation

```bash
pip install transformers peft datasets accelerate torch