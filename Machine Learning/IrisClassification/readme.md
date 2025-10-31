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
	pip install pandas numpy matplotlib scikit-learn joblib

### DataSet information:
__Features__:
  sepal.length
  sepal.width
  petal.length
  petal.width
#### Target
__variety__ :
    setosa
    versicolor
    virginica

### Workflow:
__Data Preparation__:
		•	Convert Iris.csv file into data frame
		•	Convert all column values to numeric values

__Train-Test-Split__:
		•	Split data set into 80% Training and 20% Testing set

__Pipeline Construction__:
		•	Step 1 : Standard scalar to scale all the features
		•	Step 2 : KNN,Decision Tree for result prediction

__Model Training and Evaluation__:
		•	Metrics: Accuracy,Confusion Matrix and Classification Report

__Model Saving and Loading__:
		•	Save the all models with joblib
		•	Load models for future predictions without retraining
### Running the Project:
	•	Load data set (only once)
			pandas.read_csv(file_path)
	•	Train and evaluate model:
			python3 IrisClassification.py --train
  •  Testing the model
      python3 IrisClassification.py --test
    
	•	Expected Outputs saved in_:
      Expected testing data output   :   PredictedResults.txt
  •	Visualisations
  Confusion Matrix                   :   ConfusionMatrix.png
  Classification Report              :   classification_report.txt
  feature Wise Plot                  :   featureWisePlot.png 
  
__Model Storage: Model saved as follows__
	-Trained pipeline for Decision Tree Classifier
      Model saved to path :artifact_Iris/Decision Tree Classifier.joblib
  - Trained pipeline for KNN
      Model saved to path :artifact_Iris/KNN.joblib

__Author__:
	__Vaishali M. Jorwekar__
	__Date	: 30 Oct 2025__

    

  
  

  
           


