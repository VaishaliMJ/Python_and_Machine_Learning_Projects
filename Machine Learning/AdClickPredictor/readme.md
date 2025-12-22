# 🎯 Ad Click Prediction Model

## 📌 Overview
This project implements a binary classification system to predict the probability of a user clicking on an advertisement. By utilizing **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Decision Tree** classifiers, the model identifies patterns in user behavior to optimize digital marketing strategies.

### Key Features:
*   **Automated Preprocessing:** Utilizes Scikit-Learn `Pipeline` for streamlined data workflows.
*   **Feature Scaling:** Implements `StandardScaler` to ensure numerical stability across algorithms.
*   **Multi-Model Approach:** Compares three different algorithms for optimal performance.
*   **Persistence:** Uses `joblib` for efficient model saving and loading.
*   **Visual Analysis:** Generates automated artifacts for model evaluation and feature importance.

---

## 🛠️ Dependencies
Ensure you have Python installed, then run the following to install requirements:

	pip install pandas numpy matplotlib scikit-learn joblib
---

## 📊 Dataset Information
	The model uses Ad Click Data.csv, which contains the following attributes:
###  Features:
  
Feature	Description
<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Daily Time Spent on Site</strong></td>
      <td>Average minutes a user spends on the site</td>
    </tr>
    <tr>
      <td><strong>Age</strong></td>
      <td>Customer age in years</td>
    </tr>
    <tr>
      <td><strong>Area Income</strong></td>
      <td>Average income of the user's geographical area</td>
    </tr>
    <tr>
      <td><strong>Daily Internet Usage</strong></td>
      <td>Average minutes a user spends on the internet daily</td>
    </tr>
    <tr>
      <td><strong>Ad Topic Line</strong></td>
      <td>Headline of the advertisement</td>
    </tr>
    <tr>
      <td><strong>Male</strong></td>
      <td>Gender identifier (Binary)</td>
    </tr>
    <tr>
      <td><strong>Country/City</strong></td>
      <td>Geographical information</td>
    </tr>
    <tr>
      <td><strong>Timestamp</strong></td>
      <td>Time of interaction</td>
    </tr>
  </tbody>
</table>

 
###  Target Variable
Clicked on Ad: 0 (No) or 1 (Yes)
###  ⚙️ Workflow
###  Data Preparation:<br>
1.	**Data Prep:** Data Prep :Load CSV, clean missing values, and convert categorical strings to numeric formats.
2.	**Split**: 80% Train / 20% Test split.
3.	**Pipeline**:<br>
   		* **Scale**: StandardScaler()<br>
   		* **Classify**: Fit LogisticRegression, KNeighborsClassifier, and DecisionTreeClassifier.
4.	**Evaluation**: Generate Confusion Matrix, Accuracy, and Classification Reports.
5.	**Save**: Export .joblib files for future inference.

   
####   🚀 Usage
**To Train and Evaluate**:

	python3 AdClickPrediction.py --train
**To Test Pre-trained Models**:
	
	python3 AdClickPrediction.py --test 
			
#### 📈 Visualizations & Artifacts
The following artifacts are automatically generated in the artifact_AdClickPredictor/ folder:
*	**Confusion Matrix**: visual breakdown of prediction accuracy.
*	**Feature Importance**: Plot showing which factors (e.g., Age, Income) most influence a click.
*	**Classification Report**: Detailed Precision, Recall, and F1-Score metrics.

###  Author:<br>
  Vaishali M. Jorwekar<br>
	Date	:12 Oct 2025 <br> 
  
    

  
    


     
 

