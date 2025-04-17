# 📩 Spam Message Filter – NLP Project

This is the second project of my **Natural Language Processing Internship at CodexCue**, where I built a machine learning model to classify SMS messages as either **Spam** or **Ham (Not Spam)** using Python and basic NLP techniques.

---

## 📌 Project Overview

The project detects spam messages using text classification. It involves:
- Cleaning and preprocessing SMS messages
- Converting text to numerical features (Bag of Words)
- Training a **Naive Bayes Classifier**
- Evaluating the model using accuracy, confusion matrix, and classification report

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** `pandas`, `scikit-learn`, `nltk`  
- **Model:** Multinomial Naive Bayes  
- **IDE:** Visual Studio Code

---

## 🚀 How to Run This Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/mrahim195/spam-filter.git
cd spam-filter
```

### Step 2: Create a Virtual Environment

```bash
# Windows
python -m venv venv

# macOS/Linux
python3 -m venv venv
```
### Step 3: Activate the Virtual Environment

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```
### Step 4: Install Required Libraries

```bash
pip install pandas scikit-learn nltk
```
### Step 5: Download NLTK Stopwords

```bash
import nltk
nltk.download('stopwords')
```
### Step 6: Add Dataset

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Save the file as spam.csv and place it in the project folder.

### Step 7: Run the Script

```bash
python main.py
```

### 📊 Sample Output

```bash
Accuracy: 0.988
Confusion Matrix:
[[951   4]
 [  9 151]]

Classification Report:
              precision    recall  f1-score   support
           0       0.99      1.00      0.99       955
           1       0.97      0.94      0.96       160
```

### 📁 Project Structure

```bash
spam-filter/
├── main.py
├── spam.csv
├── README.md
├── .gitignore
└── venv/
```

### 👤 About Me
Muhammad Rahim Shahid
💼 NLP Intern @ CodexCue
https://www.linkedin.com/in/muhammad-rahim-shahid-b04986268/

### 🏷️ Tags
#SpamDetection #NLP #Python #MachineLearning #ScikitLearn #TextClassification #CodexCue


