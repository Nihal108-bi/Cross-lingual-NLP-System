# README.md — Cross-lingual NLP System

## 📌 Project Overview
This project presents a Cross-lingual Natural Language Processing (NLP) system that supports:
- ✅ Bidirectional translation between English and Spanish
- ✅ Cross-lingual sentence embedding extraction using XLM-RoBERTa
- ✅ Supervised embedding alignment using Orthogonal Procrustes
- ✅ Evaluation using BLEU score and cosine similarity
- ✅ Bonus features like t-SNE visualization and a Gradio web demo

---

## 🗂️ Project Structure
```
cross_lingual_nlp/
├── data/                     # Parallel corpus files (train.en, train.es)
├── notebooks/               # Jupyter notebook pipeline
│   └── main_pipeline.ipynb
├── src/                     # Source modules
│   ├── data_loader.py
│   ├── model_handler.py
│   ├── embedding_aligner.py
│   ├── evaluation.py
│   └── web_demo.py
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🛠️ Installation
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cross_lingual_nlp.git
cd cross_lingual_nlp
```

### 2. (Optional) Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

> ⚠️ If VS Code shows unresolved imports, ensure the correct interpreter is selected via `Ctrl+Shift+P → Python: Select Interpreter`.

---

## 🚀 Usage
### 1. Run the NLP Pipeline
```bash
jupyter notebook
```
Open and run all cells in `notebooks/main_pipeline.ipynb`.

### 2. Run the Gradio Web App (Bonus)
```bash
python src/web_demo.py
```

---

## 🔍 Features Implemented
| Feature | Description |
|--------|-------------|
| ✅ Bidirectional Translation | English ↔ Spanish using Hugging Face models |
| ✅ Embedding Extraction | Using XLM-RoBERTa embeddings |
| ✅ Embedding Alignment | Supervised mapping (Procrustes method) |
| ✅ BLEU + Cosine Evaluation | sacrebleu + sklearn cosine similarity |
| ✅ t-SNE Visualization | Visual comparison of embedding spaces |
| ✅ Web Demo | Gradio interface for translation |

---

## 📈 Sample Output
```
Sample Parallel Sentences:
  EN: hello world
  ES: hola mundo

Average Cosine Similarity BEFORE Alignment: 0.9973
Average Cosine Similarity AFTER Alignment: 0.9998
BLEU (EN → ES): 85.32
BLEU (ES → EN): 91.67
Similarity (Aligned EN vs ES): 0.9969
```

---

## ❗ BLEU Score = 0.00 — Explained
Earlier, BLEU (EN → ES) returned 0.00 despite correct-looking translations.

Example:
```
REF: hola mundo
HYP: Hola mundo
```
BLEU is:
- Case-sensitive (`Hola` ≠ `hola`)
- Punctuation-sensitive (`Hola mundo.` ≠ `Hola mundo`)

### ✅ Fix: Text Normalization
We normalized all hypotheses and references:
```python
def normalize(text):
    return text.lower().strip().replace(".", "").replace(",", "")
```
This improved BLEU scores significantly.

---

## 🤖 Models Used
| Task           | Model                                      |
|----------------|---------------------------------------------|
| EN → ES        | Helsinki-NLP/opus-mt-en-es                  |
| ES → EN        | Helsinki-NLP/opus-mt-es-en                  |
| Embeddings     | xlm-roberta-base                            |

---

## 📦 Requirements
```
torch
transformers
scikit-learn
sacrebleu
gradio
numpy
matplotlib
ipykernel
sentencepiece
```

---

## 👤 Author
- **Name:** Nihal Jaiswal
- **Email:** nihaljaisawal1@gmail.com
- **GitHub:** https://github.com/yourusername

---

## 📜 License
This project was submitted as part of the **DevifyX Cross-lingual NLP Assignment** and is intended for educational use.
