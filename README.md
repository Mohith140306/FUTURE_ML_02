# 🎫 Support Ticket Classification & Priority System

> An end-to-end Machine Learning + Rule-Based Hybrid system that automatically classifies customer support tickets and assigns priority levels — helping support teams respond faster and smarter.

## 📌 Problem Statement

Customer support teams receive hundreds of tickets daily.
Manually reading, categorizing, and prioritizing each ticket is:

Slow

Inconsistent

Error-prone

Expensive

This project automates two core tasks:

- **Category Classification** — Predicts the ticket category using a trained Random Forest classifier.

- **Priority Assignment** — Assigns urgency level (High / Medium / Low) using a keyword-based scoring system.

---

## 📂 Dataset

**Source:** [Kaggle — Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

| Property | Detail |
|---|---|
| Total tickets | 8,469 |
| Categories | 5 |
| Class balance | Highly balanced (~1,634–1,752 per class) |
| Key text columns | `Ticket Subject` + `Ticket Description` |

**5 Categories:**
`Billing Issue` · Account Issue`` · `Technical Issue` · `Delivery Issuet` · `General Query`

---

## 🏗️ System Architecture

```
Raw Ticket Text
      │
      ▼
┌─────────────────────┐
│   Text Preprocessing │  lowercase → remove URLs → strip punctuation
│                     │  → remove stopwords → lemmatize
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TF-IDF Vectorizer  │  unigrams + bigrams · max 15,000 features
│                     │  min_df=2 · sublinear_tf=True
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Linear SVM Model  │  best C via GridSearchCV · class_weight=balanced
│   (LinearSVC)       │  5-fold cross-validation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌───────────────────────────┐
│  Category Prediction │────▶│  Priority Rules Engine    │
│                     │     │  keyword scoring + boost   │
└─────────────────────┘     └───────────────────────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
             { category, priority, scores }
```

---

## 🧠 Methodology

### Step 1 — Text Preprocessing
Applied NLP cleaning pipeline:

Convert text to lowercase

Remove punctuation

Remove stopwords

Lemmatization (WordNetLemmatizer)

Remove extra space
### Step 2 — Feature Engineering (TF-IDF)
TF-IDF converts text into numerical features:

Rewards important words in each ticket

Penalizes common words across all tickets

Max features: 2000

### Step 3 — Model Training
**Model**: RandomForestClassifier

Why Random Forest?

Handles non-linear patterns

Works well with medium-sized datasets

Robust to overfitting

Supports class balancing

**Configuration** :

n_estimators = 100

class_weight = 'balanced'

Train-test split: 80 / 20
### Step 4 — Hybrid Category Refinement
After ML prediction, rule-based override is applied:

Example triggers:

**Keyword**	    **Override Category**
money	       Billing & Payments
deducted	   Billing & Payments
refund	       Billing & Payments

This improves accuracy for high-risk financial issues.

### Step 5 — Priority Assignment (Rule-Based)

Priority is assigned using keyword-based scoring:
```
🔴 High

urgent

hacked

crash

money deducted

emergency

🟡 Medium

issue

problem

error

failed

🟢 Low

how

feature request

general question

```

## 📊 Model Evaluation

**Evaluation Metrics**:

Accuracy Score

Precision

Recall

F1 Score

> **Example Output:**
> Model Accuracy: 0.90
---

## 📁 Project Structure

```
support-ticket-classifier/
│
├── data/
│   └── customer_support_tickets.csv    # Raw dataset from Kaggle
│
├── notebooks/
│   └── ticket_classification.ipynb    # Data cleaning & model training
│
├── app.py                             # Interactive CLI classification tool
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/Mohith140306/
FUTURE_ML_02
cd 
FUTURE_ML_02
```

**2. Install dependencies**
```
pip install pandas nltk scikit-learn 
```

**4. Run the interactive demo**
```
python SupportTicketClassification.py
```
Type `demo` to run predictions on sample tickets, or enter any ticket text directly.

**5. Open the notebook**
```bash
jupyter notebook notebooks/ticket_classification.ipynb
```

---

## 🔧 Usage — `predict_ticket()`

```python
classify_ticket(
    "Payment failed but money deducted. Please help urgently."
) 
Output:
{
  'Category': 'Billing & Payments',
  'Priority': 'High'
}
```

---

## 💼 Business Impact

| Problem               | Solution                    | Impact                  |
| --------------------- | --------------------------- | ----------------------- |
| Manual sorting        | Automatic ML classification | Saves support team time |
| No urgency visibility | Priority tagging            | Faster response         |
| Slow ticket routing   | Hybrid refinement           | Higher accuracy         |
| Growing ticket volume | Scalable ML pipeline        | Handles large workloads |


---

## 🔮 Future Improvements

-Replace TF-IDF with Transformer embeddings (BERT)

Add confidence threshold routing

Deploy as REST API using FastAPI

Integrate with Zendesk / Freshdesk

Add multilingual support

Implement active learning loop

---
## 🛠️Technologies Used

Python

NLTK

Scikit-learn

Pandas

TF-IDF

Random Forest
## Author -

     Mohith Dappadi 

*BB.Tech AIML | Machine Learning Enthusiast*
