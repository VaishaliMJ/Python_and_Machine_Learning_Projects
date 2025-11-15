#  Overview: 
		  Built a classification model to predict passenger survival using Decision Trees and Logistic Regression 

        This project predicts whether a given passenger survived or not
          It Follows  industrial best practices  by
          	•	Automating preprocessing with Pipeline
          	•	Scaling the dataset values using StandardScalar
          	•	Used Logistic Regression and Decision Tree Classifier algorithms
          	•	Saving and loding trained model using job lib
          	•	Provided data visualisation 

##  Dependencies:
Install the required Python packages before running the project

	pip install pandas numpy matplotlib scikit-learn joblib

##  DataSet information:
Passengerid,Age,Fare,Sex,sibsp,Parch,zero,Pclass,Embarked,Survived

###  Workflow:
###  Data Preparation:
		•	Convert TitanicDataset.csv file into data frame
		•	Replace ‘0’ values in the columns by column mean() value
		•	Convert all column values to numeric values

####  Train-Test-Split:
		•	Split data set into 80% Training and 20% Testing set

####  Pipeline Construction:
		•	Step 1 : Standard scalar to scale all the features
		•	Step 2 : Used Logistic Regression and Decision Tree for result prediction

####  Model Training and Evaluation:
		•	Metrics: Accuracy,Confusion Matrix and Classification Report 
		•	Feature Importance Plot: Shows most influential features 

####  Model Saving and Loading:
		•	Save the all three models with joblib 
		•	Load models for future predictions without retraining 

####  Running the Project:
	•	Load data set (only once)
			pandas.read_csv(file_path)
	•	Train and evaluate model
			python3 TitanicPassengerSurvival.py --train
  	•	Test pretarined model
			python3 TitanicPassengerSurvival.py --test  
			
###  Expected Outputs saved:
  ####  Expected testing data output :   *PredictedResults.txt*
###  Visualisations
  Confusion Matrix                   :   *ConfusionMatrix.png*
  
  Classification Report              :   *classification_report.txt*
  
  Co-Relation Matrix                 :   *CoRelationMatrix.png* 

###  Model Storage: Model saved as follows
	-Logistic Regression : LogisticRegression.joblib
	-Decision Tree : Decision Tree Classifier.joblib

###  Author:
	 Vaishali M. Jorwekar
	 Date	:28 Sep 2025
