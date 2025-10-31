# Overview:
  Implemented classic ML classification on the Iris dataset using Decision Tree and KNN,using dataset Iris.csv<br>
  This project predicts in which category Iris flower falls<br>
          It Follows  industrial best practices  by<br>
		  
          	•	Automating preprocessing with Pipeline<br>
			
          	•	Scaling the dataset values using StandardScalar<br>
			
          	•	Used Decision Tree and KNN<br>
			
          	•	Saving and losing trained model using job lib<br>
			
          	•	Provided data visualisation<br>
			
            
 ##  Dependencies:
Install the required Python packages before running the project
	_pip install pandas numpy matplotlib scikit-learn joblib_

### DataSet information:
__Features__:<br>
  sepal.length<br>
  sepal.width<br>
  petal.length<br>
  petal.width<br>
#### Target<br>
__variety__ :<br>
    setosa<br>
    versicolor<br>
    virginica<br>

### Workflow:<br>
__Data Preparation__:<br>
		•	Convert Iris.csv file into data frame<br>
		•	Convert all column values to numeric values<br>

__Train-Test-Split__:<br>
		•	Split data set into 80% Training and 20% Testing set<br>

__Pipeline Construction__:<br>
		•	Step 1 : Standard scalar to scale all the features<br>
		•	Step 2 : KNN and Decision Tree for result prediction<br>

__Model Training and Evaluation__:<br>
		•	Metrics: Accuracy,Confusion Matrix and Classification Report<br>

__Model Saving and Loading__:<br>
		•	Save the all models with joblib<br>
		•	Load models for future predictions without retraining<br>
### Running the Project:<br>
	• **Load data set(only once)** <br>
			pandas.read_csv(file_path)<br>
	• Train and evaluate model:__<br>
			python3 IrisClassification.py --train <br>
    •  Testing the model__:<br>
      		python3 IrisClassification.py --test <br>
    
  • __Expected Outputs saved in__:<br>
      Expected testing data output   :   PredictedResults.txt<br>
  •	__Visualisations__:
  Confusion Matrix                   :   ConfusionMatrix.png<br>
  Classification Report              :   classification_report.txt<br>
  feature Wise Plot                  :   featureWisePlot.png<br>
  
__Model Storage: Model saved as follows__<br>
	-Trained pipeline for Decision Tree Classifier<br>
      Model saved to path :artifact_Iris/Decision Tree Classifier.joblib<br>
  - Trained pipeline for KNN<br>
      Model saved to path :artifact_Iris/KNN.joblib<br>

__Author__:<br>
	__Vaishali M. Jorwekar__<br>
	__Date	: 30 Oct 2025__<br>

    

  
  

  
           


