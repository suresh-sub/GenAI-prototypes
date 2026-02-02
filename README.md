# GenAI Prototypes

This repository is a collection of **hands-on GenAI experiments and prototypes**.  
The goal is to explore, build, and document practical GenAI use cases.

Each folder focuses on a **specific problem or use case** (e.g., fine-tuning, document analysis, chatbots), with code, notebooks, and notes kept as self-contained as possible.

---

## Current Projects

### 🔹 LLM Fine-tuning — Sentiment Analysis
Fine-tuned and compared two causal language models — **DistilGPT-2** and **Qwen2.5-0.5B** — for sentiment analysis using **LoRA** (parameter-efficient fine-tuning).

- Dataset: Rotten Tomatoes
- Approach: Instruction-style prompting + causal LM fine-tuning
- Libraries: Hugging Face Transformers, PEFT, Datasets
- Includes training, inference, and basic evaluation

📓 Notebook:  
[`sentiment_analysis_fine_tuning_lora.ipynb`](llm-finetune/sentiment_analysis_fine_tuning_lora.ipynb)

---

## Planned / Upcoming Areas

This repo will gradually expand to include projects such as:

- 📄 Document analysis (e.g., quarterly statements, reports)
- 🤖 GenAI chatbots (RAG, prompt engineering, tool use)

---