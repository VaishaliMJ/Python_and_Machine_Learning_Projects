# Overview 
Built Regression model to predict advertisement sales


It Follows industrial best practices by

•	Automating preprocessing with Pipeline
		
•	Scaling the dataset values using StandardScalar
		
•	Used Linear Regression algorithm
		
•	Saving and loding trained model using job lib
		
•	Provided data visualisation

## Dependencies:<br>
Install the required Python packages before running the project<br>
_pip install pandas numpy matplotlib scikit-learn joblib_
### DataSet information:<br>
#### Features and Target:<br>
TV,radio,newspaper,sales
### Workflow:

#### Data Preparation: <br>
• Convert Advertising.csv file into data frame<br>
•	Clean data set

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
	    	python3 AdvertisingSalesAnalysis.py --train<br>
        
  •	Test pre-trained model<br>
		    python3 AdvertisingSalesAnalysis.py --test<br> 

 #### Expected Outputs saved:

**Expected testing data output** : <br>
  PredictedSales.txt<br>
  MeanSquaredError.txt<br>
  

**Visualisations**<br>

  Co Realtion Matrix : CoRelationMatrix.png

  Regression Plot SalesPredictionPlot.png

**Model Storage**: Model saved as follows

  -Linear Regression.joblib<br>
    
 **Author**

 Vaishali M. Jorwekar<br>
 Date	:25 Nov 2025 
  
         

