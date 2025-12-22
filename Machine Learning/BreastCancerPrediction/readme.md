# Breast Cancer Prediction Model
## 📌 Overview
This project focuses on predicting whether a patient's tumor is **malignant** or **benign** using machine learning. It implements industrial best practices, including automated preprocessing pipelines and model persistence.

### Key Features
*   **Automated Preprocessing**: Utilizes Scikit-Learn `Pipeline` for seamless data flow.
*   **Feature Scaling**: Implements `StandardScaler` to normalize dataset values.
*   **Multi-Algorithm Approach**: Compares performance across Logistic Regression, SVM
*   **Model Persistence**: Save and load trained models using `joblib`.
*   **Comprehensive Visualization**: Includes Confusion Matrices and Feature Importance plots.
---

## 🛠 Dependencies
Ensure you have Python installed, then run the following command to install required packages:


	pip install pandas numpy matplotlib scikit-learn joblib
---

## 📊 Dataset Information
**Features**
The model utilizes 30 distinct features derived from tumor images, including:
Mean, Error, and Worst values for: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, and Fractal Dimension.<br>

**Target**<br>
*	**'target'**-malignant or benign
---
## ⚙️ Workflow
### 1.	Data Preparation:
*	Loaded data from sklearn.datasets into a DataFrame.
*	Handled missing values by dropping NA rows.
*	Ensured all columns are numeric.<br>
### 2.	Train-Test Split:
*	80% Training set
*	20% Testing set.<br>
### 3.	Pipeline Construction:
*	**Step 1**: StandardScaler for feature normalization.
*	**Step 2**:	Model implementation (Logistic Regression, SVM)<br>
### 4.	Evaluation:
*	**Metrics**: Accuracy, Confusion Matrix, Classification Report, and ROC-AUC curve<br>

## 🚀 Running the Project
1. **Train and Evaluate**<br>
	⁃	python BC_Prediction.py --train<br>
2.	**Test model**<br>
	⁃	python BC_Prediction.py --test<br>
## 📂 Outputs & Model Storage
### Visualizations & Reports
*	**ConfusionMatrix.png**: Visual representation of prediction accuracy.
*	**FeatureImportance.png**: Plot showing the most influential features.
*	**classification_report.txt**: Detailed precision, recall, and F1-score breakdown.
*	**PredictedResults.txt**: Output from the testing phase.<br>
#### Saved Models
**Trained models are stored as:**
*	Logistic Regression.joblib
*	SVM.joblib
### ✍️Author:<br>
Vaishali M. Jorwekar<br>
Date	:31 Oct 2025<br>  
  

  
  





