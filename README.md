🎫 Support Ticket Classification & Priority System

An end-to-end Machine Learning + Rule-Based Hybrid system that automatically classifies customer support tickets and assigns priority levels — helping support teams respond faster and smarter.

📌 Problem Statement

Customer support teams receive hundreds of tickets daily.
Manually reading, categorizing, and prioritizing each ticket is:

Slow

Inconsistent

Error-prone

Expensive

This project automates two core tasks:

1️⃣ Category Classification

Predicts the ticket category using a trained Random Forest classifier.

2️⃣ Priority Assignment

Assigns urgency level (High / Medium / Low) using a keyword-based scoring system.

📂 Dataset

Source: Custom labeled customer support dataset

Property	Detail
Total Tickets	~20+ (extendable)
Categories	5
Balance	Balanced sample dataset
Text Column	Ticket Description
📌 Categories

Billing Issue

Account Issue

Technical Issue

Delivery Issue

General Query

🏗️ System Architecture
Raw Ticket Text
              │
              ▼
┌───────────────────────────┐
│     Text Preprocessing    │
│ lowercase → remove        │
│ punctuation → stopwords   │
│ → lemmatization           │
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│     TF-IDF Vectorizer     │
│    max_features = 2000    │
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│ Random Forest Classifier  │
│  class_weight = balanced  │
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│   Hybrid Rule Refinement  │
│  (Billing keyword check)  │
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│    Priority Assignment    │
│   (Keyword-based logic)   │
└──────────────┬────────────┘
               │
               ▼
      { category, priority }
🧠 Methodology
Step 1 — Text Preprocessing

Applied NLP cleaning pipeline:

Convert text to lowercase

Remove punctuation

Remove stopwords

Lemmatization (WordNetLemmatizer)

Remove extra spaces

Step 2 — Feature Engineering (TF-IDF)

TF-IDF converts text into numerical features:

Rewards important words in each ticket

Penalizes common words across all tickets

Max features: 2000

Step 3 — Model Training

Model: RandomForestClassifier

Why Random Forest?

Handles non-linear patterns

Works well with medium-sized datasets

Robust to overfitting

Supports class balancing

Configuration:

n_estimators = 100

class_weight = 'balanced'

Train-test split: 80 / 20

Step 4 — Hybrid Category Refinement

After ML prediction, rule-based override is applied:

Example triggers:

Keyword	Override Category
money	Billing & Payments
deducted	Billing & Payments
refund	Billing & Payments

This improves accuracy for high-risk financial issues.

Step 5 — Priority Assignment

Priority is assigned using keyword-based scoring:

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

📊 Model Evaluation

Evaluation Metrics:

Accuracy Score

Precision

Recall

F1 Score

Example Output:

Model Accuracy: 0.90
📁 Project Structure
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

🚀 Getting Started
1️⃣ Clone the repository
git clone https://github.com/Mohith140306/FUTURE_ML_02.git
2️⃣ Install dependencies
pip install pandas nltk scikit-learn
3️⃣ Run the system
python SupportTicketClassification.py

Or open Jupyter Notebook:

jupyter notebook ticket_classification.ipynb
🔧 Usage Example
classify_ticket(
    "Payment failed but money deducted. Please help urgently."
)
Output:
{
  'Category': 'Billing & Payments',
  'Priority': 'High'
}
💼 Business Impact
Problem	                                     Solution	                            Impact
Manual sorting                          Automatic ML classification	        Saves support team time
No urgency visibility                   Priority tagging	                Faster response                                                                   
Slow ticket routing	                Hybrid refinement	                Higher accuracy
Growing ticket volume	                Scalable ML pipeline	                Handles large workloads 
🔮 Future Improvements

Replace TF-IDF with Transformer embeddings (BERT)

Add confidence threshold routing

Deploy as REST API using FastAPI

Integrate with Zendesk / Freshdesk

Add multilingual support

Implement active learning loop

🛠️ Technologies Used

Python

NLTK

Scikit-learn

Pandas

TF-IDF

Random Forest

👨‍💻 Author

Mohith Dappadi
B.Tech CSE | Machine Learning Enthusiast
