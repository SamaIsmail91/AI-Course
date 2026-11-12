

<h1 align="center">🏦 Consumer Complaint Classification</h1>
<h3 align="center">End-to-End NLP Pipeline: SimpleRNN · LSTM · GRU · DistilBERT · Gradio</h3>

<p align="center">
  <a href="https://python.org"><img src="https://shields.io" alt="Python"></a>
  <a href="https://tensorflow.org"><img src="https://shields.io" alt="TensorFlow"></a>
  <a href="https://pytorch.org"><img src="https://shields.io" alt="PyTorch"></a>
  <a href="https://huggingface.co"><img src="https://shields.io" alt="Hugging Face"></a>
  <a href="https://gradio.app"><img src="https://shields.io" alt="Gradio"></a>
  <a href="https://kaggle.com"><img src="https://shields.io" alt="Kaggle dataset"></a>
</p>

<p align="center">
  <b>🚀 <a href="https://huggingface.cospaces/samaismaill/consumer_complaint_classifier">Live Demo (Hugging Face Space)</a></b> &nbsp;|&nbsp;
  <b>📓 <a href="https://kaggle.com">Executed Notebook (Kaggle)</a></b> &nbsp;|&nbsp;
  <b>🗂️ <a href="https://kaggle.com">Dataset</a></b>
</p>

---

## 📋 Overview
An end-to-end Natural Language Processing (NLP) pipeline that automatically routes a customer's written complaint to the correct product category (`credit_card`, `credit_reporting`, `debt_collection`, `mortgages_and_loans`, `retail_banking`). It compares three recurrent networks built from scratch (**SimpleRNN**, **LSTM**, **GRU**) against a fine-tuned **DistilBERT** transformer and a classical **TF-IDF + Logistic Regression** baseline, serving the models through an interactive **Gradio** web application with real-time confidence scoring and multi-model benchmarking.

---

## 🎯 Business Value & Objectives
* **⚡ Faster Routing:** Replaces manual triage of unstructured complaint text with automated, near-instant classification.
* **💰 Lower Operational Cost:** Reduces the manual effort required to sort and route inbound tickets while minimizing human error.
* **📈 Foundation for Trend Analysis:** Enables management to monitor which financial product areas generate the highest volume of complaints and prioritize product improvements.

---

## 🗂️ Dataset Overview
**Source:** [Consumer Complaints Dataset for NLP on Kaggle](https://kaggle.com), built from complaint narratives in the public **CFPB** (U.S. Consumer Financial Protection Bureau) database.

| Property | Value |
| :--- | :--- |
| **Classes (5)** | `credit_card`, `credit_reporting`, `debt_collection`, `mortgages_and_loans`, `retail_banking` |
| **Complaints (Balanced)** | `111,670` |
| **Stratified Split** | Train: `80,681` (72%) / Validation: `14,238` (13%) / Test: `16,751` (15%) |
| **Text Field** | `narrative` (customer complaint text) |
| **Label Field** | `product` (target category) |

> **📝 Column Auto-Detection:** The notebook automatically detects text and label columns, supporting both the Kaggle schema (`narrative` / `product`) and raw CFPB exports (`Consumer complaint narrative` / `Product`). Since narratives arrive partially cleaned, preprocessing applies light normalization.

> **⚠️ No Internet / No Kaggle Access?** See `data/download_instructions.md` for manual setup, or run `data/make_sample_data.py` to generate a small synthetic dataset for smoke-testing the pipeline *(never report evaluation metrics from synthetic data)*.

---

## 🧱 End-to-End Pipeline
1. **Data Ingestion:** Load `narrative` and `product` columns.
2. **Text Preprocessing:** Lowercasing, stripping URLs, numbers, punctuation, and `XXXX` redactions, followed by stop-word removal and lemmatization.
3. **Imbalance Handling:** Majority classes undersampled to at most `3×` the size of the smallest class.
4. **Data Splitting:** Stratified `72% / 13% / 15%` train/validation/test split.
5. **Tokenization & Padding:** Keras `Tokenizer` (`20,000`-word vocabulary fit strictly on training data) with post-padding to `150` tokens.
6. **Model Training:** Custom `Embedding → Recurrent` architectures built from scratch alongside fine-tuned `DistilBERT` via Hugging Face `Trainer`.
7. **Evaluation:** Accuracy, weighted/macro precision, recall, F1-score, confusion matrices, and error analysis.
8. **Deployment:** Interactive web dashboard hosted on **Hugging Face Spaces** using **Gradio**.

---

## 🚀 Architectures & Training Setup

### Models Compared
* **SimpleRNN:** Sequential baseline built from scratch; constrained by vanishing gradients on long narratives.
* **LSTM (Long Short-Term Memory):** Gated cell states designed to retain long-range context across lengthy complaints.
* **GRU (Gated Recurrent Unit):** Streamlined gated recurrent architecture with fewer parameters and faster convergence than LSTM.
* **DistilBERT (Fine-Tuned Transformer):** Transfer learning with self-attention using `distilbert-base-uncased`.
* **Baseline (TF-IDF + Logistic Regression):** `50k` n-gram features (`1–2` grams) with `class_weight="balanced"` to benchmark deep learning gains.

### Hyperparameters & Configuration

| Model | Training Configuration |
| :--- | :--- |
| **SimpleRNN / LSTM / GRU** | `Embedding(128-d, mask_zero=True)` → `SpatialDropout(0.2)` → `Bidirectional(64)` → `Recurrent(32)` → `Dropout(0.4)` → `Dense(64, ReLU, L2)` → `Dropout(0.3)` → `Softmax`.<br>**Optimizer:** Adam (`1e-3`), batch size `64`, up to `20` epochs, Early Stopping on validation loss, `ReduceLROnPlateau`, and class weights. |
| **DistilBERT** | Max sequence length `128`, `3` epochs, batch size `16`, learning rate `2e-5`, weight decay `0.01`, mixed precision (`fp16`) on GPU, best checkpoint selected via validation F1.<br>*Note: Fine-tuned in a separate process from TensorFlow to prevent GPU memory contention.* |

---

## 📊 Evaluation & Model Benchmarking

All models were evaluated on the identical held-out test set (**16,751 complaints**). Weighted averages are reported below:

| Model Architecture | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Inference Cost |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SimpleRNN** | `72.02%` | `73.85%` | `72.02%` | `72.44%` | Very Low |
| **LSTM** | `84.64%` | `84.93%` | `84.64%` | `84.47%` | Low |
| **GRU** | `85.01%` | `85.43%` | `85.01%` | `85.08%` | Low |
| **TF-IDF + LogReg (Baseline)** | `85.80%` | — | — | `85.84%` | Ultra Low |
| **🏆 DistilBERT (Transformer)** | **`86.77%`** | **`86.76%`** | **`86.77%`** | **`86.76%`** | Moderate |

*(Full macro-averaged metrics, per-class classification reports, and confusion matrices are available in the [executed Kaggle notebook](https://kaggle.com) and `outputs/final_comparison.csv`).*

> **💡 Engineering Trade-Off:** While **DistilBERT** achieves the highest overall accuracy (`86.77%`), it outperforms the classical TF-IDF baseline by roughly `1.0%` and **GRU** by `1.76%`. For low-resource or CPU-bound production environments, **GRU** offers an optimal balance between strong predictive performance and low inference latency.

---

## 🔍 Key Findings & Error Analysis
* **Modest Transformer Gain:** A well-tuned linear model on n-gram features performs remarkably well on financial complaint routing, leaving a narrow margin for deep neural architectures.
* **Recurrent Architecture Dynamics:** Gated architectures (**GRU** and **LSTM**) substantially outperform **SimpleRNN** (`~13%` F1 boost) by mitigating vanishing gradients across multi-sentence narratives.
* **Run-to-Run Stability:** Across full training runs, accuracy variance remained under `0.4%` for LSTM, GRU, and DistilBERT, whereas SimpleRNN fluctuated by `~3.7%` (`75.7%` vs. `72.0%`). Differences below `0.5%` should be treated as statistical noise.
* **Category Overlap:** The primary source of misclassification occurs between `credit_reporting` and `debt_collection`, where customer disputes often reference both collection agencies and credit report inaccuracies simultaneously.

---

## 🧪 Methodology Notes
* **Strict Data Leakage Prevention:** Both the Keras `Tokenizer` and `TF-IDF` vectorizer are fit exclusively on the training split. Model checkpoints are selected using validation metrics, and the test set is touched once for final evaluation.
* **Class Imbalance Strategy:** Recurrent models combine undersampling with class weights; DistilBERT uses undersampling only; the baseline relies on `class_weight="balanced"`.
* **Production Distribution Caveat:** Because undersampling is applied prior to splitting, the test set is more balanced than real-world traffic where `credit_reporting` dominates. Per-class precision in live production will reflect natural class priors.

---

## 📁 Project Structure

```text
consumer_complaint_project/
├── app.py                     # Interactive Gradio web application
├── build_space.py             # Helper script to bundle artifacts for Hugging Face Spaces
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── src/
│   ├── predict.py             # Unified inference & text preprocessing pipeline
│   └── train_transformer.py   # Standalone DistilBERT fine-tuning script
├── models/                    # Saved tokenizers, label encoders, configs & trained weights
│   ├── simplernn_model.keras
│   ├── lstm_model.keras
│   ├── gru_model.keras
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   └── config.json
└── outputs/
