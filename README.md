# RajneetiDrishti - A Two-Stage Vision-Language Ensemble Framework for Political Meme Classification

<p align="center">
  <img src="Images/dataflow-meme.svg" alt="Pipeline Dataflow Animation" width="100%">
</p>

## CUET CSE Fest 2025 - National-Level Datathon

### Challenge: PoliMemeDecode — Humor That Speaks Politics

| Achievement | Ranking |
|-------------|------------|
| **Public Leaderboard** | 3rd Place |
| **Private Leaderboard** | 1st Place |
| **Overall Standing** | 4th out of 152 Teams |

**Team:** NeuronX
**Task:** Classify Bangla memes as **Political** or **Non-Political**

---

## About This Mirror

This repository is a mirror of Team NeuronX's PoliMemeDecode competition
repository, kept on my profile for portfolio purposes. The original
repository, with all of its original commits and dates, is:
https://github.com/Sayeem-Velocity/PoliMemeDecode_NeuronX

**Team NeuronX** — authors of "RajneetiDrishti: A Two-Stage Vision-Language
Ensemble Framework for Political Meme Classification", presented in the
final round of the PoliMemeDecode Datathon at CUET CSE Fest 2025:
## About This Mirror

This repository is a mirror of Team NeuronX's PoliMemeDecode competition
repository, kept on my profile for portfolio purposes. The original
repository, with all of its original commits and dates, is:
https://github.com/Sayeem-Velocity/PoliMemeDecode_NeuronX

**Team NeuronX** — authors of "RajneetiDrishti: A Two-Stage Vision-Language
Ensemble Framework for Political Meme Classification", presented in the
final round of the PoliMemeDecode Datathon at CUET CSE Fest 2025:

- S.M. Shahriar Sayeem — Chittagong University of Engineering and Technology
- Md Mobashir Hasan — Daffodil International University
- Sabit Ahamed Preanto — Daffodil International University
- Eiamin Hassan Shanto — Daffodil International University

**Result:** 3rd on the Public Leaderboard, 1st on the Private Leaderboard,
4th overall out of 152 teams (93.7% Macro F1 public, 89.6% Macro F1 private).

All other commits in this repository are the original repository's commits,
unchanged, with their original authors and dates.

- S.M. Shahriar Sayeem — Chittagong University of Engineering and Technology
- Sabit Ahamed Preanto — Daffodil International University
- Eiamin Hassan Shanto — Daffodil International University

**Result:** 3rd on the Public Leaderboard, 1st on the Private Leaderboard,
4th overall out of 152 teams (93.7% Macro F1 public, 89.6% Macro F1 private).

All other commits in this repository are the original repository's commits,
unchanged, with their original authors and dates.

## Overview

This repository contains our winning solution for the PoliMemeDecode challenge at CUET CSE Fest 2025 National-Level Datathon. We developed **RajneetiDrishti**, a two-stage Vision-Language ensemble framework for political meme classification targeting Bangladeshi political and social memes.

The task was particularly challenging due to **visually ambiguous and misleading content** that limited the effectiveness of traditional vision-based architectures. Our solution leverages state-of-the-art Vision-Language Models (VLMs) with 4-bit quantization and a static knowledge base to efficiently process and classify memes containing Bangla, English, and Banglish (Bengali text in Roman script) content.

---

## Competition Results

| Metric | Score |
|--------|-------|
| **Single Model F1-Score** | ~92% Macro F1 |
| **Ensemble F1-Score (Public)** | 93.7% Macro F1 |
| **Private Leaderboard F1-Score** | 89.6% Macro F1 (70% test data) |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Two-Stage VLM Architecture** | Qwen2.5-VL for multimodal reasoning + Phi-3-mini for final classification |
| **Static Knowledge Base** | Curated political entity database for enhanced context |
| **4-Bit Quantization** | NF4 quantization via BitsAndBytes to run on Tesla T4 (16GB VRAM) |
| **VQA Chain-of-Thought Prompting** | Structured prompting generating 5 metadata columns |
| **Multilingual OCR** | Nanonets-OCR-s for Bangla/English/Banglish text extraction |
| **7 Metadata Features** | Text + 5 VQA metadata + Language info for comprehensive analysis |
| **Memory Management** | Aggressive garbage collection to prevent CUDA OOM errors |

---

## Repository Structure

```
PoliMemeDecode_NeuronX/
├── Images/
│   ├── system architecture.png    # System architecture diagram
│   ├── reasoning.jpg              # Reasoning visualization
│   └── results.png                # Results and metrics
├── NeuronX Notebooks/
│   ├── NeuronX_Two-stage-meme-classification-qwen-phi3.ipynb
│   ├── NeuronX_qwen-vl-meme-metadata-cot.ipynb
│   └── NeuronX_nanonets-ocr-4bit-quantized.ipynb
├── NeuronX.pdf                    # Presentation slides
├── RajneetiDrishti__A_Two_Stage_Vision_Language_Ensemble_Framework_for_Political_Meme_Classification.pdf
└── README.md
```

---

## System Architecture & Workflow

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           POLIMEMEDECODE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌─────────────────────────────────────────────────────┐   │
│  │  Meme Image  │───▶│           PREPROCESSING STAGE                       │   │
│  └──────────────┘    │  nanonets/Nanonets-OCR-s                            │   │
│                      │  • Text Extraction (OCR)                             │   │
│                      │  • Language Detection (Bangla/English/Banglish)      │   │
│                      └──────────────────────┬──────────────────────────────┘   │
│                                             │                                   │
│                                             ▼                                   │
│                      ┌─────────────────────────────────────────────────────┐   │
│                      │           METADATA GENERATION                        │   │
│                      │  Qwen/Qwen2.5-VL-7B-Instruct (4-bit)                │   │
│                      │  VQA Chain-of-Thought Prompting                      │   │
│                      │  Generates 5 Metadata Columns:                       │   │
│                      │  • Humor Type • Metaphor • Meme Explanation          │   │
│                      │  • Metaphor Object • Political Intensity             │   │
│                      └──────────────────────┬──────────────────────────────┘   │
│                                             │                                   │
│                                             ▼                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                         STAGE 1: MULTIMODAL REASONING                   │    │
│  │  Qwen/Qwen2.5-VL-7B-Instruct (4-bit quantized)                         │    │
│  │  Input: Meme Image + 7 Metadata Features                                │    │
│  │         (Extracted Text + 5 VQA Metadata + Language Info)               │    │
│  │  Output: Initial Political/Non-Political Prediction                     │    │
│  └─────────────────────────────────┬──────────────────────────────────────┘    │
│                                    │                                            │
│                                    ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                         STAGE 2: FINAL CLASSIFICATION                   │    │
│  │  microsoft/Phi-3-mini-128k-instruct                                     │    │
│  │  Input: Stage 1 Predictions + Static Knowledge Base                     │    │
│  │  Output: Final Political/Non-Political Classification                   │    │
│  └─────────────────────────────────┬──────────────────────────────────────┘    │
│                                    │                                            │
│                                    ▼                                            │
│                           ┌────────────────┐                                    │
│                           │  submission.csv │                                   │
│                           └────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Preprocessing: OCR & Language Detection (nanonets/Nanonets-OCR-s)

Specialized OCR for multilingual meme text extraction:

- **Bangla Detection**: Unicode range analysis (U+0980 to U+09FF)
- **English Detection**: Dictionary-based stop word density calculation  
- **Banglish Detection**: Roman characters without English vocabulary

### Metadata Generation: VQA Chain-of-Thought (Qwen2.5-VL-7B-Instruct)

Multimodal reasoning to generate 5 metadata columns:

- **Humor Type**: Mockery, Sarcastic, Ironic, Satirical, Other
- **Metaphor Detection**: Text, Image, or Both
- **Meme Explanation**: Contextual meaning and target analysis
- **Metaphor Object**: Specific subject or entity identification
- **Political Intensity**: High, Moderate, Low classification

### Stage 1: Multimodal Reasoning (Qwen2.5-VL-7B-Instruct)

Each meme image along with **7 metadata features** is passed for multimodal reasoning:
- Extracted Text + 5 VQA Metadata Columns + Language Information

### Stage 2: Final Classification (microsoft/Phi-3-mini-128k-instruct)

Stage 1 predictions combined with a **static knowledge base** for final classification:
- Named entity recognition for politicians and parties
- Keyword matching for political terms
- Contextual validation using curated political database

---

## Notebooks Description

### 1. `NeuronX_Two-stage-meme-classification-qwen-phi3.ipynb`
**Main Classification Pipeline**

- Implements the complete two-stage classification framework
- Uses Chain-of-Thought (CoT) prompting with metadata
- Combines Qwen2.5-VL and Phi-3-Vision for robust classification
- Handles checkpoint saving and resume functionality

### 2. `NeuronX_qwen-vl-meme-metadata-cot.ipynb`
**Metadata Extraction with CoT Reasoning**

- Extracts structured metadata from meme images
- Implements few-shot Chain-of-Thought prompting
- Generates JSON output with strict schema adherence
- Handles political intensity classification with nuanced rules

### 3. `NeuronX_nanonets-ocr-4bit-quantized.ipynb`
**Multilingual OCR Pipeline**

- Specialized OCR for Bangla/English/Banglish text
- 4-bit quantized for memory efficiency (~2.5GB VRAM)
- Custom language classification heuristics
- Resolution capping for batch processing stability

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (Tesla T4 or better recommended)
- 16GB+ VRAM for full pipeline

### Installation

```bash
# Clone the repository
git clone https://github.com/Sayeem-Velocity/PoliMemeDecode_NeuronX.git
cd PoliMemeDecode_NeuronX

# Install dependencies
pip install transformers accelerate bitsandbytes
pip install pandas pillow tqdm qwen-vl-utils
```

### Running the Pipeline

1. **Upload to Kaggle** (Recommended for GPU access)
2. **Add competition dataset** to notebook inputs
3. **Run notebooks** in order:
   - OCR extraction → Metadata generation → Classification

---

## Classification Criteria

### Political Intensity Levels

| Level | Criteria |
|-------|----------|
| **HIGH** | Named politicians (Sheikh Hasina, Khaleda Zia, etc.), Political parties (AL, BNP, Jamaat), Keywords (Vote Chori, Genocide, Dictator) |
| **MODERATE** | General institutional criticism, Social satire without specific names, Cultural figures |
| **LOW** | Daily life content, Pop culture, Sports, Relationships |

---

## Technical Specifications

| Component | Specification |
|-----------|---------------|
| **OCR Model** | nanonets/Nanonets-OCR-s (4-bit NF4) |
| **Metadata & Stage 1 Model** | Qwen/Qwen2.5-VL-7B-Instruct (4-bit NF4) |
| **Stage 2 Model** | microsoft/Phi-3-mini-128k-instruct |
| **Quantization** | BitsAndBytes NF4 with double quantization |
| **Compute Dtype** | bfloat16 / float16 |
| **Max Tokens** | 512 (configurable) |
| **Target GPU** | Tesla T4 16GB |

---

## Results & Leaderboard

![Results](Images/results.png)

### Performance Summary

| Configuration | Macro F1-Score | Notes |
|--------------|----------------|-------|
| Single Model (Qwen2.5-VL) | ~92% | Without ensembling |
| Ensemble (Public LB) | **93.7%** | 30% test data |
| Private Leaderboard | **89.6%** | 70% test data |

### System Architecture

![System Architecture](Images/system%20architecture.png)

### Static Knowledge Base

![Static Knowledge Base](Images/static%20knowledge%20base.png)

### Stage 2 Validation

![Stage 2 Validation](Images/stage%202%20validation.png)

### Reasoning Visualization

![Reasoning](Images/reasoning.jpg)

### Error Analysis

![Error Analysis](Images/error%20analysis.png)

### Improvement Analysis

![Improvement](Images/improvement.png)

---

## Documentation

- [NeuronX Presentation (PDF)](NeuronX.pdf)
- [Full Paper: RajneetiDrishti Framework](RajneetiDrishti__A_Two_Stage_Vision_Language_Ensemble_Framework_for_Political_Meme_Classification.pdf)

---

## Team NeuronX

**CUET CSE Fest 2025 - National-Level Datathon**

**Achievements:**
- 3rd Place on Public Leaderboard
- 1st Place on Private Leaderboard  
- 4th Overall out of 152 Teams

*Hard luck on the final standing, but it was a great learning experience! Looking forward to achieving the best in upcoming challenges.*

---

## License

This project is developed for academic competition purposes.

---

## Acknowledgments

- CUET CSE Fest 2025 Organizers
- Hugging Face for Transformers library
- Qwen team for Vision-Language models
- Microsoft for Phi-3 Vision model
- Nanonets for OCR model

---

## Contact

For questions or collaboration, please open an issue in this repository.
