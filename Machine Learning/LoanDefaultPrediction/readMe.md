# 🏦 Loan Default Prediction System
## 📌 Project Overview
The goal of this project is to build a machine learning model that predicts the likelihood of a borrower defaulting on a loan based on historical data. By accurately identifying high-risk applicants, financial institutions can minimize financial losses and optimize their lending strategies.

## 📊 Dataset Description
The project uses the Loan_default.csv dataset, which contains demographic, financial, and behavioral attributes of borrowers.
*  **Target Variable**: Loan_Default (1 = Defaulted, 0 = Paid)
*  **Key Features**:
  *  Financials: Income, Loan Amount, Credit Score, Debt-to-Income (DTI) ratio.
  *  Demographics: Age, Employment Status, Home Ownership.
  *  Behavioral: Previous loan defaults, Number of credit lines.

---
## 🛠️ Tech Stack
*  **Language**: Python 3.9+
*  **Libraries**:
  *  **Data Handling**: Pandas, NumPy
  *  **Visualization**: Matplotlib, Seaborn
  *  **Modeling**: Scikit-learn, Gradient Boosting , Random Forest

## 📈 Methodology
*  **Exploratory Data Analysis (EDA)**: Visualizing feature distributions and correlations
*  **Preprocessing**: Handling missing values, encoding categorical variables (Label Encoding), and scaling numerical  data.
*  **Model Training**: Training multiple classifiers including Random Forest and Gradient Boosting.
---
## ⚙️ Workflow

#### Data Preparation:

*  **Data Prep** :Load CSV, clean missing values, and convert categorical strings to numeric formats.
*  **Split**: 80% Train / 20% Test split.
*  **Pipeline**:
    * Scale: StandardScaler()
    * Classify: Fit Random Forest,Gradient Boosting and Voting Classifier
*  **Evaluation**: Generate Confusion Matrix, Accuracy, and Classification Reports.
*  **Save**: Export .joblib files for future inference.
---
### 💻 Running the Project:

*  **Train Model**:
    Train the Model To load the data, train the algorithm, and generate visualizations:<br>
      python3 LoanDefaultPrediction.py --train<br>
*  **Test Model**:
    Test the Model using command:<br>
      python3 LoanDefaultPrediction.py --test<br>
---
### 📊 Expected Outputs saved:

The following insights are automatically saved to the project directory:

*  **Co-Relation.png**: Feature dependency heatmap.
*  **AccuracyPlot.png**: Accuracy of models.
*  **ConfusionMatrix.png**: Confusion Matrix.
*  **DataAnalysis.png**: Data Analysis as per features.
*  **DefaulterAnalysis**: A summary of dafulter.
*  **PredictedResult.txt**: Predicted Test Results
*  **classification_report.txt**: Classification Report<br>

### 💿 Model File:

*  Model saved in ARTIFACT_LoanDefault folder sutomatically

---
#### 👩‍💻 Author
Vaishali M. Jorwekar<br>
Date :21 Dec 2025
  




        
