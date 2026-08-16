# Student Performance Prediction using Machine Learning

A machine learning project that predicts student academic performance (Low, Medium, or High) based on behavioral and demographic factors, using classification algorithms in Python.

## 📌 Overview
This project analyzes the **xAPI-Edu-Data dataset** (480 student records) to identify which factors — such as raised hands, visited resources, discussion participation, and absence days — most influence student academic performance. Multiple classifiers are trained and compared to find the most accurate prediction model.

## 📊 Dataset
- **Source:** Students' Academic Performance Dataset (xAPI-Edu-Data)
- **Records:** 480
- **Features:** Gender, Nationality, Grade Level, Semester, Raised Hands, Visited Resources, Announcements Viewed, Discussion, Parent Survey, Absence Days, etc.
- **Target:** Class (L = Low, M = Medium, H = High)

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** scikit-learn, pandas, NumPy, Seaborn, Matplotlib

## 🤖 Algorithms Used
Trained and compared 5 classification models:
- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression
- Perceptron
- Multi-Layer Perceptron (MLP) Neural Network

## ⚙️ Methodology
1. Data preprocessing — label encoding of categorical features
2. 70/30 train-test split
3. Model training across all 5 classifiers
4. Evaluation using precision, recall, F1-score, and accuracy
5. Exploratory data visualization (Seaborn/Matplotlib) to identify trends across attendance, grade, and gender

## 📈 Results
The models were evaluated and compared based on classification accuracy to determine the most effective algorithm for predicting student performance.

## 🚀 How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
python Project.py
```

## 📁 Project Structure
```
├── Project.py        # Main script with data preprocessing, model training & evaluation
├── AI-Data.csv        # Dataset
└── README.md
```
