# Diabetes Prediction Health Classification System
Developed a health classification system using medical data to predict diabetes using Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree classifiers.
## 🚀Overview: 
This project predicts whether a patient is diabetic or non-diabetic based on the diabetes.csv dataset. It follows industrial best practices by:		 
*	**Automating Preprocessing**: Utilizing Scikit-Learn Pipeline.
*	**Feature Scaling**: Standardizing dataset values using StandardScaler.
*	**Model Persistence**: Saving and loading trained models using joblib.
*	**Data Visualization**: Providing insights through confusion matrices and feature importance plots.


## 🛠 Dependencies:
Install the required Python packages before running the project

	pip install pandas numpy matplotlib scikit-learn joblib

## 📊  DataSet information:
###  Features:
	1.	Pregnancies
	2.	Glucose
	3.	BloodPressure
	4.	SkinThickness
	5.	Insulin
	6.	BMI 
	7.	DiabetesPedigreeFunction
	8.	Age

###  Target:
	Outcome 0 :  Non Diabetic
	Outcome 1 :  Diabetic

---
	
### ⚙️ Workflow:
### 1.	Data Preparation:
*	Convert diabetes.csv into a Pandas DataFrame.
*	Replace '0' values in physiological columns with the column mean() to handle missing data.
*	Convert all feature columns to numeric values.

#### 2.	Train-Test-Split:
*	Split the dataset into 80% Training and 20% Testing sets

#### 3.	Pipeline Construction:
*	**Step 1**: StandardScaler to scale all features for uniform distribution.
*	**Step 2**: Implementation of Logistic Regression, KNN, and Decision Tree classifiers.


#### 4.	Model Training and Evaluation:
*	**Metrics**: Accuracy,Confusion Matrix and Classification Report 
*	**Visuals**: Feature Importance Plot: Shows most influential features 

#### 5.	Model Saving and Loading:
*	Models are serialized using joblib for future predictions without retraining.
----  
####  🏃Running the Project:
*	**Load and Train the models**:
		python3 DiabetesPrediction.py --train
*	**Test pretarined model**:
		python3 DiabetesPrediction.py --test  
			
###  📁 Project Artifacts
#### Expected Outputs saved:
*	**PredictedResults.txt**: Results from the testing phase.
*	**ConfusionMatrix.png**: Visual representation of prediction accuracy.
*	**classification_report.txt**: Detailed precision, recall, and F1-score metrics.
*	**FeatureImportance.png**: Plot showing the most influential features.


###  Model Storage: 
Trained models are saved in the artifact_Diabetes/ directory:
*	Logistic Regression :LogisticRegression.joblib
*	KNN : KNN.joblib
*	Decision Tree : Decision Tree Classifier.joblib

###  ✍️Author:
	 Vaishali M. Jorwekar
	 Date	:12 Oct 2025
