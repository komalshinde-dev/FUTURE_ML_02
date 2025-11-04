# 🧮 FUTURE INTERNS – MACHINE LEARNING TASK 2  
## 💡 Churn Prediction System

### 🎯 Project Objective
Develop a **machine learning model** that predicts which customers are likely to leave a telecom service.  
This project helps businesses understand the key factors influencing customer churn and take preventive action.

---

## 📦 Dataset

**Source:** [Telco Customer Churn Dataset (IBM)](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

- Contains customer demographics, service details, and churn status.  
- Target column: `Churn` (`Yes` = Customer left, `No` = Customer retained).  

File used in project: `churn.csv`

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| Language | Python 3 |
| Libraries | pandas, numpy, scikit-learn, seaborn, matplotlib |
| Algorithm | Random Forest Classifier |
| Environment | VS Code / Google Colab |

---

## 🧠 Skills Gained
- Data Preprocessing & Cleaning  
- Label Encoding and Feature Scaling  
- Classification Model Building  
- Model Evaluation (Metrics & Confusion Matrix)  
- Feature Importance Analysis  
- Data Visualization with Seaborn & Matplotlib  

---

## 🧩 Project Workflow

1. **Data Loading & Cleaning**  
   - Load the dataset, convert data types, and handle missing values.  

2. **Encoding & Scaling**  
   - Encode categorical variables using `LabelEncoder` and normalize features with `StandardScaler`.  

3. **Model Training**  
   - Train a `RandomForestClassifier` on the processed data.  

4. **Evaluation & Insights**  
   - Calculate accuracy, classification report, and visualize confusion matrix.  
   - Plot feature importance to understand key churn drivers.  

---

## 🚀 How to Run This Project

### 🔹 Step 1 – Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
