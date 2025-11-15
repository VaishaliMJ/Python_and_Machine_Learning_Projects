# Overview 
Built Classification models to analyse the correlation between brain size and Intelligence


It Follows industrial best practices by

•	Automating preprocessing with Pipeline
		
•	Scaling the dataset values using StandardScalar
		
•	Used Decision Tree Classifier algorithm
		
•	Saving and loding trained model using job lib
		
•	Provided data visualisation

## Dependencies:<br>
Install the required Python packages before running the project<br>
pip install pandas numpy matplotlib scikit-learn joblib

### DataSet information:<br>
#### Features and Target:<br>
Gender,Age Range,Head Size(cm^3),Brain Weight(grams)<br>



### Workflow:

#### Data Preparation: <br>
• Convert MarvellousHeadBrain.csv file into data frame<br>
•	Convert all column values to numeric values

#### Pipeline Construction:<br>
  •	Step 1 : Standard scalar to scale all the features<br>
	•	Step 2 : Used Decision Tree for result prediction
  
#### Model Training and Evaluation:<br>
  •	Metrics: Accuracy,Confusion Matrix and Classification Report<br>
	•	Feature Importance Plot: Shows most influential features 

#### Model Saving and Loading:<br>
  •	Save the model with joblib <br>
	•	Load models for future predictions without retraining<br>

#### Running the Project:<br>
  •	Load data set (only once)<br>
		    pandas.read_csv(file_path)<br>
  •	Train and evaluate model<br>
	    	python3 HeadBrainClassification.py --train<br>
  •	Test pretarined model<br>
		    python3 HeadBrainClassification.py --test<br>  

#### Expected Outputs saved:

**Expected testing data output** : <br>
  PredictedResults.txt

**Visualisations**<br>

  Confusion Matrix : ConfusionMatrix.png

  Classification Report : classification_report.txt

  Co Realtion Matrix : CoRelationMatrix.png

**Model Storage**: Model saved as follows

  -Decision Tree Classifier.joblib<br>
    
 **Author**

 Vaishali M. Jorwekar<br>
 Date	:10 Oct 2025 
  
  
  
