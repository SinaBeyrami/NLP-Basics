# NLP-Basics: Persian/English Movie EDA & Genre Classification

A single Jupyter notebook that performs **EDA**, **text preprocessing** (Persian & English), **visualization**, **outlier analysis**, **class balancing**, and **fine-tuning BERT** models for **genre classification** from plot summaries.

* Notebook: `AI_NLP_Project_Spring2024_Student_Version.ipynb`
* Dataset: `persianmovies.csv` (sample rows shown at the end of this README)

## Features

* Dual-language preprocessing:

  * **English**: spaCy lemmatization, HTML removal, punctuation/number stripping, stopword removal, ASCII normalization.
  * **Persian**: HAZM normalization, character unification, diacritic removal, HTML removal, tokenization, stopword removal, stemming.
* Rich EDA:

  * Descriptive stats, histograms/densities, categorical bar/pie charts.
  * Genre analysis (word clouds & frequent terms per genre), time trends, rating analysis.
  * Correlations, pair plots, PCA.
  * Outlier detection (boxplots + IQR report).
* Class balancing:

  * Simple thresholded over/undersampling (in-notebook) **and** a SMOTE experiment.
* Modeling:

  * **Persian:** `HooshvareLab/bert-fa-zwnj-base` (preprocessed vs. raw).
  * **English:** `bert-base-uncased` (preprocessed vs. raw).
  * Evaluation with **Accuracy, F1 (weighted)**, and **Confusion Matrix**.

---

## Quickstart

### 1) Environment

Python 3.9+ recommended; GPU strongly recommended.

```bash
# create env (optional)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install "transformers[torch]" torch torchvision torchaudio \
            hazm python-bidi wordcloud seaborn scikit-learn \
            spacy beautifulsoup4 imbalanced-learn matplotlib
python -m spacy download en_core_web_sm
```

> The notebook also calls `nltk.download('punkt')` and `nltk.download('stopwords')` at runtime.

### 2) Data

Place the CSV at: `data/persianmovies.csv`
Columns (original → renamed in notebook):

* `Content_1` → **Persian summary**
* `Content_2` → **English summary**
* plus: `Link, EN_title, PENGLISH_title, PERSIAN_title, Score, Year, Genre, Time`

> The current notebook loads from a Windows absolute path. Change this line:
>
> ```python
> df = pd.read_csv("data/persianmovies.csv")
> ```
>
> The renaming step is already in the notebook.

### 3) Run

Open the notebook and execute cells top to bottom. On Colab, select **Runtime → Change runtime type → GPU**.

---

## What the notebook does (section map)

1. **Preprocessing**

   * Builds `Preprocessed English` and `Preprocessed Persian`.
   * Handles HTML, punctuation, numbers/specials, stopwords, lemmatization/stemming, non-ASCII cleanup, Persian character normalization.

2. **Descriptive statistics & visualization**

   * Numeric stats (Score/Year/Time), unique counts (Genre/titles).
   * Histograms, density plots, bar/pie charts, relationship plots, correlation heatmap & pair plots.

3. **Genre analysis**

   * Word cloud for genre distribution.
   * Top-10 frequent words **by genre** on both raw and preprocessed summaries (Persian & English).
   * Genre vs. (Score/Year/Time) via boxplots.

4. **Time & rating analysis**

   * Releases over years; Year vs. Score/Time lines.
   * Rating distribution; frequent words by rating bins; Score vs. Year/Time.

5. **Multivariate & PCA**

   * Heatmap correlations; pair plots.
   * PCA (2D) colored by Score, Year, or Time.

6. **Outliers**

   * Boxplots + IQR detection; removes four `Score==0.0` rows as data issues.

7. **Balancing & feature engineering**

   * Maps long-tail genres → `New_Genre` (fewer base classes).
   * Train/test split (80/20), simple thresholded resampling, label & one-hot encoders.
   * Class distribution plots before/after resampling.

8. **Fine-tuning**

   * Persian BERT (preprocessed and raw).
   * English BERT (preprocessed and raw).
   * Metrics and confusion matrices.
   * **SMOTE** experiment (English, preprocessed).

---

## Results (high level)

* With naive resampling, **test accuracy/F1 ≈ 0.5** (varies per run/hardware).
* Applying **SMOTE** on the English preprocessed run improved to **\~0.62 accuracy / \~0.59 F1** in the notebook.
* Preprocessing (especially stopwords/normalization) noticeably improves top-words interpretability and often stabilizes training.

> Note: Using SMOTE directly on `input_ids` is a *rough* proxy and can be brittle; see “Improvements” below.

---

## Reproducibility tips

* Set seeds (NumPy/Torch/Random) near the top of the notebook:

  ```python
  import random, numpy as np, torch
  SEED = 42
  random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
  ```
* Keep the **test split untouched** by any resampling.
* Log metrics via `Trainer`’s `compute_metrics` for consistent reporting.

---

## Known gotchas & quick fixes

* **Absolute path**: Replace the Windows path with `data/persianmovies.csv`.
* **spaCy model**: Must run `python -m spacy download en_core_web_sm` once locally.
* **Persian word clouds**: WordCloud needs a Unicode font. Use:

  ```python
  WordCloud(font_path="fonts/Vazirmatn-Regular.ttf", width=800, height=400, background_color="white")
  ```

  (Any TTF that supports Persian/Arabic script works.)
* **Function name typo**: `Englsih_Preprocessor` → consider renaming to `EnglishPreprocessor`.
* **Tokenizers name clash**: You import `word_tokenize` from NLTK *and* HAZM; prefer:

  ```python
  from hazm import word_tokenize as hazm_tokenize
  from nltk.tokenize import word_tokenize as nltk_tokenize
  ```

  and call them explicitly to avoid surprises.
* **SMOTE caution**: SMOTE on token IDs is not linguistically meaningful. Prefer alternatives below.

---

## Suggested improvements (if you iterate)

1. **Class imbalance**

   * Use `Trainer` with **class weights** (weighted cross-entropy) or **WeightedRandomSampler**.
   * Try **focal loss** or **logit adjustment**.
   * Try **text-level augmentation** (EDA, synonym replacement, back-translation) before tokenization.

2. **Modeling**

   * Add a **validation split** separate from test; set `evaluation_strategy="steps"` and early-stop on validation F1.
   * Log `compute_metrics` to track Accuracy/F1 during training:

     ```python
     def compute_metrics(eval_pred):
         logits, labels = eval_pred
         preds = logits.argmax(-1)
         return {
             "accuracy": accuracy_score(labels, preds),
             "f1_weighted": f1_score(labels, preds, average="weighted"),
         }
     ```
   * Try **Multilingual models** (e.g., `bert-base-multilingual-cased`) for a single joint model.

3. **Features**

   * Include simple **metadata** (Year/Time) alongside text using a small MLP head or late fusion.

4. **Efficiency**

   * Use `DataCollatorWithPadding`, lower `max_length` (e.g., 256), and enable mixed precision (`fp16=True`) if GPU supports it.

---

## Sample of `data/persianmovies.csv`

```
Link,EN_title,PENGLISH_title,PERSIAN_title,Content_1,Content_2,Score,Year,Genre,Time
https://.../local-anaesthetic-bi-hessie-mozeie,Local Anaesthetic,Bi Hessie Mozeie,بی‌حسی موضعی,"جلال‌، دانشجوی سابق...", "Jalal, a dropouts philosophy student...",4.8,2018,Drama,73
https://.../disturbance-ashoftegi,Disturbance,Ashoftegi,آشفته گی,"«آشفته‌گی» رئالیستی...", "After the murder of his rich twin brother...",3.8,2018,Crime,78
...
```

---

## Authors

* Sina Beyrami (400105433)
* Aren GolAzizian (99171366)
* M. Hossein HajiHosseini (99101427)

---

## License

Academic/educational use.

---
