# Overview 
Built Regression models to analyse the correlation between brain size and Intelligence


It Follows industrial best practices by

•	Automating preprocessing with Pipeline
		
•	Scaling the dataset values using StandardScalar
		
•	Used Linear Regression algorithm
		
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
	•	Step 2 : Used Linear Regression for result prediction
  
#### Model Training and Evaluation:<br>
  •	Metrics: Mean squared error,Root mean squared error,R^2 <br>
  •	Co-Relation Matrix

#### Model Saving and Loading:<br>
  •	Save the model with joblib <br>
	•	Load models for future predictions without retraining<br>

#### Running the Project:<br>
  •	Load data set (only once)<br>
		    pandas.read_csv(file_path)<br>
  •	Train and evaluate model<br>
	    	python3 HeadBrainRegression.py --train<br>
  •	Test pretarined model<br>
		    python3 HeadBrainRegression.py --test<br>  

#### Expected Outputs saved:

**Expected testing data output** : <br>
  PredictedBrainWeight.txt<br>
  MeanSquaredError.txt<br>
  

**Visualisations**<br>

  Co Realtion Matrix : CoRelationMatrix.png

  Regression Plot HeadSizeVsBrainWtRegressionPlot.png

**Model Storage**: Model saved as follows

  -Linear Regression.joblib<br>
    
 **Author**

 Vaishali M. Jorwekar<br>
 Date	:8 Oct 2025 
  
  
  
