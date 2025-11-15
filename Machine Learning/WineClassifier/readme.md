# Overview 
Implmented ML models for Multi-class classification of Wine Type using KNN and Random Forest

It Follows industrial best practices by

•	Automating preprocessing with Pipeline
		
•	Scaling the dataset values using StandardScalar
		
•	Used Random Forest and KNN
		
•	Saving and losing trained model using job lib
		
•	Provided data visualisation

## Dependencies:<br>
Install the required Python packages before running the project<br>
pip install pandas numpy matplotlib scikit-learn joblib

### DataSet information:<br>
#### Features:<br>
Alcohol,Malic acid,Ash,Alcalinity of ash,Magnesium,
Total phenols,Flavanoids,Nonflavanoid phenols,Proanthocyanins,
Color intensity,Hue,OD280/OD315 of diluted wines,Proline<br>
#### Target<br>
Class


### Workflow:

#### Data Preparation: <br>
• Convert WinePredictor.csv file into data frame<br>
•	Convert all column values to numeric values

#### Pipeline Construction:<br>
  •	Step 1 : Standard scalar to scale all the features<br>
	•	Step 2 : Used KNN and Random Forest for result prediction
  
#### Model Training and Evaluation:<br>
  •	Metrics: Accuracy,Confusion Matrix and Classification Report<br>
	•	Feature Importance Plot: Shows most influential features 

#### Model Saving and Loading:<br>
  •	Save the all two models with joblib <br>
	•	Load models for future predictions without retraining<br>

#### Running the Project:<br>
  •	Load data set (only once)<br>
		    pandas.read_csv(file_path)<br>
  •	Train and evaluate model<br>
	    	python3 WinePredictor.py --train<br>
  •	Test pretarined model<br>
		    python3 WinePredictor.py --test<br>  

#### Expected Outputs saved:

**Expected testing data output** : <br>
  PredictedResults.txt

**Visualisations**<br>

  Confusion Matrix : ConfusionMatrix.png

  Classification Report : classification_report.txt

  Feature Importance : FeatureImportance.png

**Model Storage**: Model saved as follows

  -Random Forest:Random Forest Classifier.joblib<br>
  -KNN : KNN.joblib
    
 **Author**

 Vaishali M. Jorwekar<br>
 Date	:21 Oct 2025 
  
  
  
