# 🎯 Ad Click Prediction Model

## 📌 Overview
This project aims to predict the probability of a user clicking on an advertisement. This assignment uses  **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Decision Tree** classification algorithms. 

### Key Features:
*   **Preprocessing:** Uses Scikit-Learn `Pipeline` for sequential work flow.
*   **Scaling:** Implements `StandardScaler` to ensure all numerical values are in range
*   **Different Models** Compare three different algorithms for performance.
*   **Model Saving and Loading:** Uses `joblib` for efficient model saving and loading.
*   **Visual Analysis:** Generates automated artifacts for model evaluation and feature importance.

---

## 🛠️ Dependencies
Ensure you have Python installed, with the following libraries:

	pip install pandas numpy matplotlib scikit-learn joblib
---

## 📊 Dataset Information
	The model uses "Ad Click Data.csv", with the following attributes:
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
      <td>Gender</td>
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
1.	**Data Prep:** : Load CSV, clean missing values, and convert strings to numeric formats.
2.	**Split**: Split the data set into 80% Train and  20% Test data set.
3.	**Pipeline**:<br>
   		* **Scale**: StandardScaler()<br>
   		* **Algorithms**: Fit LogisticRegression, KNeighborsClassifier, and DecisionTreeClassifier.
4.	**Evaluation**: Generate Confusion Matrix, Accuracy, and Classification Reports.
5.	**Save**: Use .joblib files for future inference.

   
####   🚀 Usage
**To Train and Evaluate**:

	python3 AdClickPrediction.py --train
**To Test Pre-trained Models**:
	
	python3 AdClickPrediction.py --test 
			
#### 📈 Visualizations 
The following artifacts are automatically generated in the **artifact_AdClickPredictor/** folder:
*	**Confusion Matrix**: Confusion matrix with visual display and comparison for all algorithms.
*	**Feature Importance**: Plot the feature importance of all features.
*	**Classification Report**: Detailed Precision, Recall, and F1-Score metrics for all algorithms in 	   "classificationReport.txt" file.

###  ✍️Author:<br>
  Vaishali M. Jorwekar<br>
	Date:12 Oct 2025 <br> 
  
    

  
    


     
 

