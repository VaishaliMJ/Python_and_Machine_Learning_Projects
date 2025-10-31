# Overview:
  Implemented classic ML classification on the Iris dataset using Decision Tree and KNN,using dataset Iris.csv<br>
  This project predicts in which category Iris flower falls<br>
          It Follows  industrial best practices by
		  
          	•	Automating preprocessing with Pipeline
			
          	•	Scaling the dataset values using StandardScalar
			
          	•	Used Decision Tree and KNN
			
          	•	Saving and losing trained model using job lib
			
          	•	Provided data visualisation
			
            
 ##  Dependencies:
Install the required Python packages before running the project
	_pip install pandas numpy matplotlib scikit-learn joblib_

### DataSet information:
__Features__: <br>
  	- sepal.length <br>
  	- sepal.width <br>
  	- petal.length<br>
  	- petal.width<br>
#### Target<br>
__variety__ :<br>
    -	setosa<br>
    -	versicolor<br>
    -	virginica<br>

### Workflow:<br>
__Data Preparation__:
		•	Convert Iris.csv file into data frame
		
		•	Convert all column values to numeric values

__Train-Test-Split__:

		•	Split data set into 80% Training and 20% Testing set

__Pipeline Construction__:

		•	Step 1 : Standard scalar to scale all the features
		•	Step 2 : KNN and Decision Tree for result prediction

__Model Training and Evaluation__:
		•	Metrics: Accuracy,Confusion Matrix and Classification Report
		
		

__Model Saving and Loading__:

		•	Save the all models with joblib
		•	Load models for future predictions without retraining
		
### Running the Project:

	• Load data set(only once) 
			pandas.read_csv(file_path)
			
	• Train and evaluate model:
			python3 IrisClassification.py --train
			
    •  Testing the model:
      		python3 IrisClassification.py --test 
    
  • __Expected Outputs saved in__:
  
      Expected testing data output   :   PredictedResults.txt
	  
  •	__Visualisations__:
  
  		Confusion Matrix                   :   ConfusionMatrix.png
	
  		Classification Report              :   classification_report.txt
	
  		feature Wise Plot                  :   featureWisePlot.png
  
__Model Storage: Model saved as follows__

	-	Trained pipeline for Decision Tree Classifier
			Model saved to path :artifact_Iris/Decision Tree Classifier.joblib   
   	-	Trained pipeline for KNN<br>
      		Model saved to path :artifact_Iris/KNN.joblib<br>

__Author__:<br>
	__Vaishali M. Jorwekar__<br>
	__Date	: 30 Oct 2025__<br>

    

  
  

  
           


