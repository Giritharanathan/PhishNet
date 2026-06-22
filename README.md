# 🎣 PhishNet

PhishNet is a machine learning-powered email classifier that detects phishing attempts before they reel you in. Built with Scikit-learn, it analyzes email text, URLs, and keyword patterns to label messages as **Phishing** or **Safe** — with confidence scores.

---

## 🚀 Demo

```
Email  : urgent click here verify account suspended http://fake-site.xyz
Result : 🚨 Phishing (97.0% confidence)

Email  : team lunch is at noon see you at the usual restaurant
Result : ✅ Safe (88.0% confidence)
```

---

## 🧠 How It Works

PhishNet combines two types of signal:

| Signal | What it looks at |
|--------|-----------------|
| TF-IDF | Word & phrase frequency across the email body |
| URL Analysis | Suspicious TLDs (`.xyz`, `.tk`, `.ml`, `.pw`) |
| Keyword Flags | Urgency words: *verify, suspended, act now, click* |

These are fed into a **Random Forest Classifier** (100 trees) trained on a labeled dataset of phishing and legitimate emails.

---

## 📦 Setup

```bash
pip install scikit-learn pandas numpy
python mailguard.py
```

---

## 📊 Results

```
Accuracy : 90.00%

Confusion Matrix:
  True Positive  = 4   (Phishing caught)
  False Positive = 0   (Legit flagged as phishing)
  False Negative = 1   (Phishing missed)
  True Negative  = 5   (Legit correctly cleared)
```

---

## 📁 Structure

```
PhishNet/
├── mailguard.py    # Core model — train, evaluate, predict
└── README.md
```

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat-square&logo=pandas)

---

## 💡 Use Case

Perfect for understanding how production spam filters work at the core — feature extraction, vectorization, ensemble classifiers, and evaluation metrics — all in under 100 lines of Python.

---

*Built as part of a Cybersecurity ML mini project.*
