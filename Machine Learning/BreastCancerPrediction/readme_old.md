# Overview<br>
Achieved high accuracy using Logistic Regression & Support Vector Machine(SVM)<br>
 This project predicts whether a given patient tumor is maligent or benign<br>
      It Follows  industrial best practices  by<br>
      	•	Automating preprocessing with Pipeline<br>
      	•	Scaling the dataset values using StandardScalar<br>
      	•	Used Logistic Regression & Support Vector Machine(SVM) algorithms<br>
      	•	Saving and loding trained model using job lib<br>
      	•	Provided data visualisation <br>
## Dependencies<br>
Install the required Python packages before running the project<br>
_**pip install pandas numpy matplotlib scikit-learn joblib**_
## DataSet information:
**Features**<br>
	  mean radius,mean texture,mean perimeter,mean area,mean smoothness,mean compactness ,mean concavity,mean concave points 
	  mean symmetry,mean fractal dimension,radius error,texture error,perimeter error,area error,smoothness error,compactness error
	  concavity error,concave points error,symmetry error,fractal dimension error,worst radius,worst texture,worst perimeter 
	  worst area,worst smoothness,worst compactness,worst concavity,worst concave points,worst symmetry,worst fractal dimension<br>
**Target** :'target'<br>malignant or benign<br>
      
**Workflow**<br>
**Data Preparation**:
	•	Loaded data from sckitlearn datasets into data frame<br>
	•	drop 'na' values<br>
	•	Convert all column values to numeric values <br>
  
  **Train-Test-Split:** <br>
	•	Split data set into 80% Training and 20% Testing set<br>
	**Pipeline Construction**:<br>
	•	Step 1 : Standard scalar to scale all the features<br>
	•	Step 2 : Used Random Forest,Decision Tree and Gradient Boosting for result prediction<br>

**Model Training and Evaluation**<br>
	•	Metrics: Accuracy,Confusion Matrix and Classification Report,ROC-AUC curve<br>
	•	Feature Importance Plot: Shows most influential features<br>
**Model Saving and Loading**<br>
	•	Save the all three models with joblib<br>
	•	Load models for future predictions without retraining <br> 
**Running the Project**<br>
	•	Load data set (only once)<br>
	•	Train and evaluate model:<br>
	⁃		python BC_Prediction.py --train<br>
  •	Test model<br>
	⁃		python BC_Prediction.py --test<br>
**Expected Outputs saved**<br>

**Expected testing data output** : PredictedResults.txt<br>

**Visualisations**

  Confusion Matrix : ConfusionMatrix.png

  Classification Report : classification_report.txt

  Feature Importance : FeatureImportance.png<br><br>
**Model Storage: Model saved as follows**<br>
  Logistic Regression.joblib and SVM.joblb
  
**Author:**<br>
 Vaishali M. Jorwekar<br>
 Date	:31 Oct 2025<br>  
  

  
  





