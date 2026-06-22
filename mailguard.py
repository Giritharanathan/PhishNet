import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import re
import warnings
warnings.filterwarnings("ignore")


PHISHING_EMAILS = [
    ("urgent account verify login immediately suspended click here http://fake-bank.xyz/login", 1),
    ("congratulations you won lottery claim prize now http://scam-prize.net/claim", 1),
    ("your paypal account limited please verify identity http://paypa1-secure.com", 1),
    ("dear customer your amazon order suspended update payment http://amaz0n-billing.tk", 1),
    ("security alert unusual signin detected verify account http://google-security.ml", 1),
    ("IRS tax refund pending confirm details http://irs-refund-2024.xyz", 1),
    ("your bank account blocked enter credentials http://chase-secure-login.net", 1),
    ("netflix payment failed update billing info http://netflix-billing-update.com", 1),
    ("microsoft account compromised reset password immediately http://msft-reset.pw", 1),
    ("free iphone winner selected click claim reward http://free-gifts-now.tk", 1),
    ("verify your email address or account will be deleted http://confirm-identity.xyz", 1),
    ("your package held customs pay fee release http://dhl-customs-fee.net", 1),
    ("crypto investment guaranteed returns 500 percent click http://crypto-profits.ml", 1),
    ("dear user your social security number suspended http://ssa-verify-now.com", 1),
    ("login attempt blocked confirm your identity http://fb-security-check.xyz", 1),
    ("urgent wire transfer required ceo authorization needed offshore account", 1),
    ("your password expired reset now http://outlook-password-reset.tk", 1),
    ("winner notification you have been selected prize claim http://winner-notify.net", 1),
    ("account verification required 24 hours or permanent deletion http://verify-acct.xyz", 1),
    ("investment opportunity guaranteed profit contact now hidden wealth", 1),
]

LEGIT_EMAILS = [
    ("team meeting scheduled for friday 3pm conference room bring your laptops", 0),
    ("quarterly report attached please review before thursday presentation", 0),
    ("reminder project deadline next week please update your tasks in jira", 0),
    ("happy birthday wishing you a wonderful day filled with joy", 0),
    ("your order has shipped estimated delivery tuesday tracking number provided", 0),
    ("new research paper published machine learning advances 2024 conference", 0),
    ("lunch plans changed moving to 1pm at the usual place downtown", 0),
    ("code review comments added to your pull request please address feedback", 0),
    ("budget approved for q3 projects team can proceed with planned roadmap", 0),
    ("newsletter monthly digest technology trends industry updates best practices", 0),
    ("interview scheduled for thursday 10am virtual meeting link attached", 0),
    ("your subscription renewal confirmation receipt attached for records", 0),
    ("book recommendation just finished great novel thought you would enjoy", 0),
    ("sprint retrospective notes attached action items assigned to team members", 0),
    ("annual performance review reminder complete self assessment by friday", 0),
    ("flight confirmation booking reference number itinerary details attached", 0),
    ("welcome to the team onboarding documents first day instructions hr", 0),
    ("meeting notes from yesterday action items owners deadlines listed below", 0),
    ("software update release notes changelog version improvements bug fixes", 0),
    ("thank you for attending conference feedback survey link enclosed", 0),
]

EXTRA_PHISHING = [
    ("act now limited time offer exclusive deal money back guarantee click", 1),
    ("your account needs immediate attention verify personal information", 1),
    ("suspicious activity detected confirm login credentials security alert", 1),
    ("you have unclaimed funds government grant apply free money now", 1),
    ("cheap medications no prescription required order now discreet shipping", 1),
]

EXTRA_LEGIT = [
    ("client proposal document version two review comments welcome", 0),
    ("reminder gym class starts at 7am see you there workout", 0),
    ("project update backend api completed frontend integration in progress", 0),
    ("family reunion planning tentative date august photos from last year", 0),
    ("conference call agenda items to discuss quarterly goals team updates", 0),
]

all_data = PHISHING_EMAILS + LEGIT_EMAILS + EXTRA_PHISHING + EXTRA_LEGIT
df = pd.DataFrame(all_data, columns=["text", "label"])


def extract_url_features(text):
    urls = re.findall(r'http[s]?://\S+', text)
    suspicious_tlds = ['.xyz', '.tk', '.ml', '.pw', '.net', '.cc']
    has_suspicious_url = int(any(any(tld in url for tld in suspicious_tlds) for url in urls))
    url_count = len(urls)
    return has_suspicious_url, url_count


def build_feature_matrix(texts):
    features = []
    for text in texts:
        has_sus_url, url_count = extract_url_features(text)
        urgent_words = sum(1 for w in ['urgent', 'immediately', 'verify', 'suspended', 'alert', 'limited', 'act now', 'click'] if w in text.lower())
        features.append([has_sus_url, url_count, urgent_words])
    return np.array(features)


X_text = df["text"]
y = df["label"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
X_test_tfidf = tfidf.transform(X_test_text).toarray()

X_train_extra = build_feature_matrix(X_train_text)
X_test_extra = build_feature_matrix(X_test_text)

X_train_final = np.hstack([X_train_tfidf, X_train_extra])
X_test_final = np.hstack([X_test_tfidf, X_test_extra])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Safe", "Phishing"])

print("=" * 50)
print("   Phishing Email Detection Model Results")
print("=" * 50)
print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
print("Confusion Matrix:")
print(f"  TP={cm[1][1]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TN={cm[0][0]}")
print("\nClassification Report:")
print(report)


def classify_email(email_text):
    tfidf_feat = tfidf.transform([email_text]).toarray()
    extra_feat = build_feature_matrix([email_text])
    combined = np.hstack([tfidf_feat, extra_feat])
    prediction = model.predict(combined)[0]
    confidence = model.predict_proba(combined)[0].max() * 100
    label = "Phishing" if prediction == 1 else "Safe"
    return label, confidence


print("\n" + "=" * 50)
print("   Sample Predictions")
print("=" * 50)

test_samples = [
    "urgent click here verify account suspended http://fake-site.xyz",
    "team lunch is at noon see you at the usual restaurant",
    "your paypal limited update billing info http://paypa1.tk",
    "happy to share the meeting notes from yesterday call",
]

for sample in test_samples:
    label, conf = classify_email(sample)
    print(f"\nEmail : {sample[:60]}...")
    print(f"Result: {label} ({conf:.1f}% confidence)")
