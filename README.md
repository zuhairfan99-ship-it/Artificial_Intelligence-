# Email Spam or Not Spam Classification

A beginner-friendly NLP and machine-learning project that predicts whether pasted email text is **Spam** or **Not Spam**. It uses TF-IDF for text features, Logistic Regression for classification, and Streamlit for the user interface.

## Features
- Included labelled CSV dataset (`data/emails.csv`)
- Text-to-number conversion using TF-IDF
- Logistic Regression classifier
- Train/test split and evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Streamlit web application
- Model is trained automatically the first time the app is launched

## Project structure

```text
spam_email_classifier/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── emails.csv
├── models/
│   └── (trained model is saved here)
└── screenshots/
    └── (add your own application screenshots here)
```

## 1. Requirements
- Python 3.9 or newer
- Internet connection for installing Python packages
- VS Code or another Python editor

## 2. Setup

Open a terminal in this project folder.

Create and activate a virtual environment (recommended):

**Windows PowerShell**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

## 3. Train and evaluate the model
```bash
python train_model.py
```
The script prints accuracy, precision, recall, F1-score, and a confusion matrix. It saves the trained model in `models/spam_classifier.joblib`.

## 4. Run the application
```bash
streamlit run app.py
```
A local browser tab should open. Paste email text and click **Check email**.

## 5. Dataset information
The included `data/emails.csv` contains 120 short, manually prepared example messages with two columns:
- `text`: the email/message content
- `label`: `spam` or `ham` (`ham` means not spam)

**Important limitation:** This is a small starter dataset for demonstrating the complete workflow. Its evaluation score is not evidence of real-world email-filtering performance. For a stronger project, replace it with a larger, legally usable email corpus such as the Apache SpamAssassin Public Corpus:
- https://spamassassin.apache.org/old/publiccorpus/

After obtaining a dataset, prepare a CSV with the same `text` and `label` columns and labels `spam` and `ham`, then rerun `python train_model.py`.

## 6. GitHub upload
Create a new GitHub repository, then run these commands inside the project folder:

```bash
git init
git add .
git commit -m "Build email spam classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own repository URL. Do not upload your virtual environment. The `.gitignore` file excludes it.

## 7. Deploy on Streamlit Community Cloud
1. Push the project to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app** and select your repository and branch.
4. Set the main file path to `app.py`.
5. Click **Deploy** and wait for the app to build.
6. Copy the public app URL for your submission.

## 8. Screenshots for submission
Run the app locally and capture:
1. The empty application screen.
2. A spam prediction result.
3. A not-spam prediction result.

Save the screenshots in the `screenshots/` folder before submitting.

## 9. Concepts used
- **NLP:** techniques for processing human language text.
- **TF-IDF:** gives numerical importance scores to words and phrases.
- **Logistic Regression:** a supervised classification algorithm.
- **Accuracy:** overall proportion of correct predictions.
- **Precision:** how many emails predicted as spam were actually spam.
- **Recall:** how many actual spam emails were correctly detected.
- **F1-score:** harmonic mean of precision and recall.

## Disclaimer
This project is for learning and demonstration. A message predicted as not spam may still be harmful, and a spam prediction can be wrong. Do not rely on this small model for security or real inbox protection.
