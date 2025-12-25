# Breast Cancer Prediction System
## 📌 Overview
This project focuses on predicting whether a tumor is **malignant** or **benign** using machine learning. It implements industrial best practices as follows
### Key Features
*   **Preprocessing**: Uses Scikit-Learn `Pipeline` automating process.
*   **Scaling**: Implements `StandardScaler` to normalize numerical dataset values.
*   **Classification Algorithms**: Compares performance across Logistic Regression, SVM
*   **Model Saving and loading**: Save and load trained models using `joblib`.
*   **Visualization**: Includes Confusion Matrices and Feature Importance plots.
---

## 🛠 Dependencies
Ensure Python installed, then run the following command to install required libraries:


	pip install pandas numpy matplotlib scikit-learn joblib
---

## 📊 Dataset Information
**Features**
The dataset has 30 distinct features derived from tumor images, including:
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
*	**ConfusionMatrix.png**: Visual representation of Confusion Matrix and comparison
*	**FeatureImportance.png**: Plot showing the most importatnt features.
*	**classification_report.txt**: For all algorithms precision, recall, and F1-score are saved in text file.
*	**PredictedResults.txt**: Output from the testing model.<br>
#### Saved Models
**Trained models are stored as:**
*	Logistic Regression.joblib
*	SVM.joblib
### ✍️Author:<br>
Vaishali M. Jorwekar<br>
Date	:31 Oct 2025<br>  
  

  
  





