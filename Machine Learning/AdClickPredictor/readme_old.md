#  Overview: 
This project Classification model predicting probability of user clicks on advertisement
"Ad Click Prediction" using Logistic Regression,KNN and Decision Tree classifier

This project predicts whether a given user will click on the Ad or not using “Ad Click Data.csv”
  It Follows  industrial best practices  by<br>
      •	Automating preprocessing with Pipeline<br>
      •	Scaling the dataset values using StandardScalar<br>
      •	Used Logistic Regression,KNN and Decision Tree Classifier algorithms<br>
      •	Saving and loding trained model using job lib<br>
      •	Provided data visualisation<br><br>
##  Dependencies:
Install the required Python packages before running the project<br>
      **_pip install pandas numpy matplotlib scikit-learn joblib_**<br>
##  DataSet information:
###  Features:
  •  Daily Time Spent on Site<br>
  •  Age<br>
  •  Area Income<br>
  •  Daily Internet Usage<br>
  •  Ad Topic Line<br>
  •  City<br>
  •  Male<br>
  •  Country<br>
  •  Timestamp<br>
 
###  Target
Clicked on Ad<br><br>
###  Workflow:
###  Data Preparation:<br>
  •	Convert Ad Click Data.csv file into data frame<br>
	•	Clean data set<br>
	•	Convert all column values to numeric values<br><br>
####  Train-Test-Split:<br>
  •	Split data set into 80% Training and 20% Testing set<br>

####  Pipeline Construction:<br>
  •	Step 1 : Standard scalar to scale all the features<br>
  •	Step 2 : Used Logistic Regression,KNN,Decision Tree for result prediction<br>
####  Model Training and Evaluation:<br>
  •	Metrics: Accuracy,Confusion Matrix and Classification Report<br> 
  •	Feature Importance Plot: Shows most influential features<br> 

####  Model Saving and Loading:<br>
  •	Save the all three models with joblib<br>
	•	Load models for future predictions without retraining<br> 
####  Running the Project:<br>
  •	Load data set (only once)<br>
			pandas.read_csv(file_path)<br>
  •	Train and evaluate model<br>
			python3 AdClickPrediction.py --train<br>
  •	Test pretarined model<br>
			python3 AdClickPrediction.py --test<br>  
			
###  Expected Outputs saved:
  ####  Expected testing data output :   *PredictedResults.txt*
###  Visualisations : Saved at folder artifact_AdClickPredictor
  Confusion Matrix                   :   *ConfusionMatrix.png*
  
  Classification Report              :   *classification_report.txt*
  
  Feature Importance                 :   *FeatureImportance.png* 

###  Model Storage: Model saved as follows<br>
  -Logistic Regression : artifact_AdClickPredictor/LogisticRegression.joblib<br>
  -KNN : artifact_AdClickPredictor/KNN.joblib<br>
  -Decision Tree : artifact_AdClickPredictor/Decision Tree Classifier.joblib<br>

###  Author:<br>
  Vaishali M. Jorwekar<br>
	Date	:12 Oct 2025 <br> 
  
    

  
    


     
 
