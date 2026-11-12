<h1 align="center">🏦 Consumer Complaint Classification</h1> <h3 align="center">End-to-End NLP Pipeline: SimpleRNN · LSTM · GRU · DistilBERT · Gradio</h3> <p align="center"> <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python"></a> <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-Keras-orange.svg" alt="TensorFlow"></a> <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-Transformers-red.svg" alt="PyTorch"></a> <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg" alt="Hugging Face"></a> <a href="https://gradio.app/"><img src="https://img.shields.io/badge/Gradio-Deployment-green.svg" alt="Gradio"></a> <a href="https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp"><img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg" alt="Kaggle dataset"></a> </p> <p align="center"> <b>🚀 <a href="https://huggingface.co/spaces/samaismaill/consumer_complaint_classifier_nlp">Live Demo (Hugging Face Space)</a></b> &nbsp;|&nbsp; <b>📓 <a href="https://www.kaggle.com/code/samaismail91/consumer-complaint-classifier-nlp>">Executed Notebook (Kaggle)</a></b> &nbsp;|&nbsp; <b>🗂️ <a href="https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp">Dataset</a></b> </p>

An end-to-end Natural Language Processing (NLP) pipeline that automatically routes a customer's written complaint to the correct product category (credit card, credit reporting, debt collection, mortgages and loans, retail banking). It compares three recurrent networks built from scratch (SimpleRNN, LSTM, GRU) against a fine-tuned DistilBERT and a classical TF-IDF + Logistic Regression baseline, and serves the models through an interactive Gradio web app that returns the predicted category with a confidence score.

🎯 Business Value & Objectives
Faster routing: replaces manual triage of unstructured complaint text with automated, near-instant classification.
Lower operational cost: reduces the manual effort needed to sort and route inbound complaints and helps reduce human error.
Foundation for trend analysis: once complaints are categorized automatically, management can monitor which product areas generate the most complaints and prioritize product improvements.
🗂️ Dataset Overview

Source: Consumer Complaints Dataset for NLP on Kaggle, built from complaint narratives in the public CFPB (U.S. Consumer Financial Protection Bureau) complaint database.

Property	Value
Classes (5)	credit_card, credit_reporting, debt_collection, mortgages_and_loans, retail_banking
Complaints after class balancing	111,670
Stratified split	train 80,681 / validation 14,238 / test 16,751 (72% / 13% / 15%)
Text field	narrative (complaint text)
Label field	product (category)

The notebook detects the text/label columns automatically, so it also works with the original CFPB column names (Consumer complaint narrative / Product). The narratives in this dataset arrive partly cleaned (lowercase, no punctuation), so the cleaning step changes them only slightly.

⚠️ No internet / no Kaggle access? See data/download_instructions.md for manual download, or run data/make_sample_data.py to generate a small synthetic file with the same structure. Synthetic data is for smoke-testing the pipeline only; never report metrics from it.

🧱 Pipeline
Data – load the narrative and product columns.
Preprocessing – lowercase, strip URLs / numbers / punctuation / XXXX redactions, remove stop words, lemmatize.
Class imbalance – majority classes undersampled to at most 3× the smallest class.
Split – stratified train / validation / test.
Sequences – Keras Tokenizer (20,000-word vocabulary, fit on the training split only) and post-padding to 150 tokens.
Models – Embedding → recurrent stacks built from scratch; DistilBERT through the Hugging Face Trainer.
Evaluation – accuracy, precision, recall, F1, confusion matrices, error analysis.
Deployment – Gradio app on Hugging Face Spaces.
🚀 Architectures Compared

To find the right balance between predictive power and production cost, four architectures plus a classical baseline were trained and benchmarked:

SimpleRNN – sequential baseline built from scratch; limited by vanishing gradients on long sequences.
LSTM (Long Short-Term Memory) – gated cell states to capture longer context in lengthy complaint narratives.
GRU (Gated Recurrent Unit) – a lighter variant of LSTM with fewer parameters and faster training.
Fine-tuned Transformer (DistilBERT) – transfer learning with self-attention via Hugging Face (distilbert-base-uncased).
Baseline: TF-IDF + Logistic Regression – 50k features, 1–2-grams, class_weight="balanced"; included to show what deep learning actually adds.

Training setup

Model	Configuration
SimpleRNN / LSTM / GRU	Embedding (128-d, mask_zero=True) → SpatialDropout (0.2) → Bidirectional layer (64) → recurrent layer (32) → Dropout (0.4) → Dense (64, ReLU, L2) → Dropout (0.3) → softmax. Adam (1e-3), batch 64, up to 20 epochs, early stopping on validation loss, LR reduction on plateau, class weights
DistilBERT	max length 128, 3 epochs, batch 16, learning rate 2e-5, weight decay 0.01, best checkpoint chosen by validation F1, mixed precision on GPU. Fine-tuned in a separate process from TensorFlow, because sharing one process and GPU between the two frameworks can exhaust GPU memory or crash the kernel
📊 Evaluation & Model Benchmarking

All models were evaluated on the same held-out test set (16,751 complaints). Weighted averages are shown:

Model Architecture	Test Accuracy	Precision (Weighted)	Recall (Weighted)	F1-Score (Weighted)	Inference cost
SimpleRNN	72.02 %	73.85 %	72.02 %	72.44 %	Very low
LSTM	84.64 %	84.93 %	84.64 %	84.47 %	Low
GRU	85.01 %	85.43 %	85.01 %	85.08 %	Low
DistilBERT (Transformer)	86.77 %	86.76 %	86.77 %	86.76 %	Moderate

The TF-IDF + Logistic Regression baseline reached 85.80 % accuracy (85.84 % weighted F1) on the same split. Macro-averaged scores, per-class reports and confusion matrices are in the executed notebook and in outputs/final_comparison.csv.

💡 Engineering trade-off: DistilBERT gives the highest accuracy, but only about one point above the TF-IDF baseline and about 1.8 points above GRU. GRU remains a strong choice for low-resource deployments because it is much cheaper to run than a transformer.

Key findings
DistilBERT is the best model, but the margin is small. A simple linear model on n-gram features is already strong on this task; the transformer adds a modest gain.
The from-scratch networks do not beat the baseline. GRU and LSTM are slightly below it, and SimpleRNN is far behind, consistent with plain RNNs struggling to carry information across long sequences.
Run-to-run variation exists. Between two full training runs, accuracy moved by less than 0.4 points for LSTM, GRU and DistilBERT, but by about 3.7 points for SimpleRNN (75.7 % vs 72.0 %). Gaps below about half a point between models should not be over-interpreted.
Errors come from overlapping categories. The largest confusion is between credit_reporting and debt_collection; many misclassified complaints (for example disputes about a collection account written as credit-report complaints) are genuinely ambiguous, so accuracy well above this level is unlikely without cleaner labels.
🧪 Methodology Notes (read before comparing numbers)
No data leakage: the tokenizer and the TF-IDF vectorizer are fit on the training split only; checkpoints are chosen on validation; the test set is used for the final comparison. The model ranking was read off the test set, so treat small gaps as indicative.
Imbalance handling differs by model: SimpleRNN / LSTM / GRU use undersampling and class weights; DistilBERT uses undersampling only; the baseline uses class_weight="balanced".
The test set is not the real-world distribution: undersampling happens before the split, so the test set is more balanced than real incoming complaints. credit_reporting is the dominant class in practice, so per-class precision in production will differ.
Train / inference consistency: every model, including DistilBERT, is trained and served on the cleaned text. Feeding raw text to the transformer at inference would silently lower real-world accuracy.
Padding mask: the Embedding layer uses mask_zero=True so recurrent layers ignore padding. Without it, LSTM/GRU process dozens of padding steps after short texts and stayed at chance accuracy in early smoke tests.
📐 Math Behind the Models (short version)
Softmax: p_k = exp(z_k) / Σ_j exp(z_j); cross-entropy loss: L = −Σ_k y_k log p_k.
LSTM: gates f_t, i_t, o_t = σ(W [h_{t−1}, x_t] + b); cell c_t = f_t ⊙ c_{t−1} + i_t ⊙ tanh(W_c [h_{t−1}, x_t] + b_c); output h_t = o_t ⊙ tanh(c_t).
GRU merges the forget and input gates into an update gate z_t: h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t, giving fewer parameters and faster training.
SimpleRNN suffers from vanishing gradients on long sequences, which motivates gated cells.
TF-IDF weights a term by tf × log(N / df) (with sublinear tf); Logistic Regression is a linear softmax classifier on those features.
🎨 Interactive Gradio Web App

An interactive UI deployed with Gradio lets support agents and engineers test live predictions. It features:

A text box for pasting a complaint narrative.
A model selector to switch between the trained architectures on the fly.
The predicted category with a confidence score (softmax probability).
A horizontal bar chart of the top-3 most probable categories.

👉 Try it: huggingface.co/spaces/samaismaill/consumer_complaint_classifier_nlp

<p align="center"> <img src="outputs/gradio_screenshot.png" alt="Gradio app interface" width="850"> </p>
📁 Project Structure
consumer_complaint_project/
├── notebooks/
│   └── Consumer_Complaint_Classification.ipynb   # complete end-to-end workflow in one notebook
├── data/
│   ├── download_instructions.md   # how to get the dataset
│   └── make_sample_data.py        # synthetic data generator (smoke tests only)
├── src/
│   ├── preprocessing.py           # cleaning, imbalance handling, tokenization, padding, splits
│   ├── models_rnn.py              # SimpleRNN / LSTM / GRU architectures
│   ├── train_deep_models.py       # training and checkpointing of the recurrent models
│   ├── train_transformer.py       # DistilBERT fine-tuning with the HF Trainer
│   ├── evaluate.py                # metrics, confusion matrices, model comparison
│   └── predict.py                 # unified inference used by the Gradio app
├── models/                        # trained weights, tokenizer, label encoder (not committed; served from the Space)
├── outputs/                       # comparison table, confusion matrices, error analysis, app screenshot
├── app.py                         # Gradio application
├── build_space.py                 # packages the files a Hugging Face Space needs into hf_space/
├── requirements.txt
└── README.md
⚙️ Installation & Execution
Option 1 — Kaggle (recommended)
New Notebook → File → Import notebook and upload notebooks/Consumer_Complaint_Classification.ipynb.
Add Input → add the dataset; the notebook finds the CSV automatically.
Settings: Accelerator = GPU, Internet = On (needed for pip and for downloading DistilBERT).
Save Version → Save & Run All (Commit) keeps all outputs.
Option 2 — Locally
bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4 punkt
# place the dataset CSV in data/ (see data/download_instructions.md), then either open the notebook or run:
python src/preprocessing.py && python src/train_deep_models.py && python src/train_transformer.py && python src/evaluate.py
python app.py        # launches the Gradio interface locally
☁️ Deploy to Hugging Face Spaces

Kaggle's /kaggle/working is temporary, so push the trained artifacts to the Hub from the notebook (the Back up the trained artifacts cell: set PUSH_TO_HUB = True, your username and Space name, and add an HF_TOKEN secret), or download them from the committed version's Output tab.

models/ must contain tokenizer.pkl, label_encoder.pkl, config.json, versions.json, the *.keras files and optionally transformer_model/ (config.json, model.safetensors, tokenizer files).
Run python build_space.py (or python build_space.py --code-only if the models are already on the Space). It creates hf_space/ with app.py, src/predict.py, a Space README.md and a requirements.txt pinned to the library versions recorded in versions.json (the pickled tokenizer and the .keras files are version-sensitive).
Upload the contents of hf_space/ to the Space, keeping the src/ and models/ folders. The Space needs its own README.md with the YAML configuration header generated by build_space.py, so do not overwrite it with this file.
⚠️ Limitations & Future Work
English only; trained on CFPB-style narratives and may not generalize to other domains.
The confidence score is the softmax probability, not a calibrated probability.
Results come from single training runs; the test distribution is balanced rather than real-world; class labels are coarse and overlap (especially credit_reporting vs debt_collection).
Future work: multiple seeds with significance testing, probability calibration, hyperparameter tuning, and training DistilBERT on lightly cleaned (not stop-word-stripped) text.
🙏 Acknowledgements

Data: Consumer Complaints Dataset for NLP on Kaggle, derived from the public CFPB complaint database. Built with TensorFlow/Keras, PyTorch, Hugging Face Transformers, scikit-learn

