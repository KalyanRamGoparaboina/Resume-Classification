import os
import pandas as pd
import re
from pathlib import Path
import docx
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle

def extract_text(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    text = ""
    try:
        if suffix == ".docx":
            doc = docx.Document(file_path)
            text = " ".join([p.text for p in doc.paragraphs])
        elif suffix == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        elif suffix == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text.strip()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    base_dir = Path(r"c:\Users\gopar\OneDrive\Desktop\Resume\Resume classification dataset\Dataset\Resumes")
    data = []
    
    categories = {
        "Peoplesoft resumes": "Peoplesoft",
        "SQL Developer Lightning insight": "SQL Developer",
        "workday resumes": "Workday"
    }
    
    for folder in base_dir.iterdir():
        if folder.is_dir():
            cat_name = categories.get(folder.name, folder.name)
            for file in folder.rglob("*"):
                if file.suffix.lower() in [".docx", ".pdf", ".txt"]:
                    text = extract_text(file)
                    if text:
                        data.append({"file_name": file.name, "text": text, "category": cat_name})
        else:
            if folder.suffix.lower() in [".docx", ".pdf", ".txt"]:
                text = extract_text(folder)
                if text:
                    label = "React Developer" if "React" in folder.name else "Other"
                    data.append({"file_name": folder.name, "text": text, "category": label})

    df = pd.DataFrame(data)
    if df.empty:
        print("No data extracted.")
        return

    df['cleaned_text'] = df['text'].apply(clean_text)
    
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['category_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, kernel='linear', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    
    performance_metrics = []
    best_acc = 0
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        performance_metrics.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })
        
        print(f"\n{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    # Save performance CSV
    perf_df = pd.DataFrame(performance_metrics)
    project_dir = Path(r"c:\Users\gopar\OneDrive\Desktop\Resume\project")
    perf_df.to_csv(project_dir / "model_performance.csv", index=False)

    # Save the best model
    with open(project_dir / "model.pkl", "wb") as f:
        pickle.dump(models[best_model_name], f)
    with open(project_dir / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(project_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
        
    df.to_csv(project_dir / "processed_resumes.csv", index=False)
    print(f"\nTraining Complete. Best Model: {best_model_name}")

if __name__ == "__main__":
    main()
