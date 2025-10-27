# H1 Overview: Developed a health classification system using medical data to predict diabetes
          Diabetes Prediction using Logistic Regression and Decision Tree classifier

        This project predicts whether a given patient is diabetic  or non-diabetic using “diabetes.csv”
          It Follows  industrial best practices  by
          	•	Automating preprocessing with Pipeline
          	•	Scaling the dataset values using StandardScalar
          	•	Used Logistic Regression as classification algorithm
          	•	Saving and losing trained model using job lib
          	•	Provided data visualisation 

## H2 Dependencies:
Install the required Python packages before running the project
	pip install pandas numpy matplotlib scikit-learn joblib

## H2 DataSet information:
### H3 Features:
	1.	Pregnancies
	2.	Glucose
	3.	BloodPressure
	4.	SkinThickness
	5.	Insulin
	6.	BMI 
	7.	DiabetesPedigreeFunction
	8.	Age

### H3 Target:
	#### H4 Outcome 
			0 : Non Diabetic
			1:  Diabetic
### H3 Workflow:
### H4 Data Preparation:
		•	Convert .csv file into data frame
		•	Replace ‘0’ values in the columns by column mean() value
		•	Convert all column values to numeric values

#### H4 Train-Test-Split:
		•	Split data set into 80% Training and 20% Testing set

#### H4 Pipeline Construction:
		•	Step 1 : Standard scalar to scale all the features
		•	Step 2 : Used Logistic Regression,KNN,Decision Tree for result prediction

#### H4 Model Training and Evaluation:
		•	Metrics: Accuracy,Confusion Matrix and Classification Report
		•	Feature Importance Plot: Shows most influential features

#### H4 Model Saving and Loading:
		•	Save the all three models with joblib
		•	Load models for future predictions without retraining

#### H4 Running the Project:
	•	Load data set (only once)
			pandas.read_csv(file_path)
	•	Train and evaluate model:
			python3 DiabetesPrediction.py
### H3 Expected Outputs saved in :
  #### H4 Expected testing data output :   PredictedResults.txt
### H3 Visualisations
  Confusion Matrix                   :   ConfusionMatrix.png
  Classification Report              :   classification_report.txt
  Feature Importance                 :   FeatureImportance.png 

### H3 Model Storage: Model saved as follows
	-Logistic Regression : artifact_Diabetes/LogisticRegression.joblib
	-KNN : artifact_Diabetes/KNN.joblib
	-Decision Tree : artifact_Diabetes/Decision Tree Classifier.joblib

### H3 Author:
	Vaishali M. Jorwekar
	Date	:12 Oct 2025
