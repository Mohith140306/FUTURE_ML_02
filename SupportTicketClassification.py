import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# --- 1. SETUP ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- 2. ENHANCED CATEGORY LOGIC ---
def refine_category(text, ml_predicted_category):
    """
    Overrides ML prediction if specific high-intent keywords are found.
    """
    billing_triggers = ['money', 'deducted', 'payment', 'charge', 'refund', 'transaction', 'billing', 'invoice']
    technical_triggers = ['crash', 'bug', 'install', 'software', 'update', 'frozen']
    
    text_lower = text.lower()
    
    # Check for Billing intent first (Financial issues are high priority)
    if any(word in text_lower for word in billing_triggers):
        return "Billing & Payments"
    
    # If no billing keywords, stick with the ML prediction
    return ml_predicted_category

# --- 3. PRIORITY LOGIC ---
def assign_priority(text):
    high_keywords = ['urgent', 'crash', 'money', 'deducted', 'security', 'hack', 'emergency']
    text = text.lower()
    if any(word in text for word in high_keywords):
        return "High"
    return "Low/Medium"

# --- 4. DATA PROCESSING & TRAINING ---
def clean_text(text):
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

try:
    df = pd.read_csv('customer_support_tickets.csv')
    df['cleaned_description'] = df['Ticket Description'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=2000)
    X = tfidf.fit_transform(df['cleaned_description']).toarray()
    y = df['Ticket Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Adding class_weight='balanced' helps the model not ignore smaller categories
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model trained with balanced weights!")

except FileNotFoundError:
    print("Please ensure 'customer_support_tickets.csv' is in the folder.")

# --- 5. INTERACTIVE TEST ---
print("\n--- Support Ticket System (Hybrid Mode) ---")
while True:
    user_input = input("\nEnter ticket (or 'exit'): ")
    if user_input.lower() == 'exit': break
    
    # ML Prediction
    cleaned = clean_text(user_input)
    vec = tfidf.transform([cleaned]).toarray()
    raw_ml_prediction = model.predict(vec)[0]
    
    # Hybrid Refinement
    final_category = refine_category(user_input, raw_ml_prediction)
    priority = assign_priority(user_input)
    
    print(f"\nResult for: '{user_input}'")
    print(f"Final Category: {final_category}")
    print(f"Priority: {priority}")