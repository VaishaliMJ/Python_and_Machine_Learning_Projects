# 🧠 Brain Size & 🤖Intelligence Classification
This project implements a machine learning classification model to analyze the correlation between brain size and intelligence, utilizing industrial best practices for automation and scalability.
## 📌 Overview 
Built Classification models to analyse the correlation between brain size and Intelligence

## 🚀 Key Features
*	**Automated Preprocessing**: Utilizes Scikit-Learn Pipeline for seamless data transformation.
*	**Feature Scaling**: Implements StandardScaler to normalize dataset values.
*	**Decision Tree Classifier**: Leverages tree-based learning for predictive analysis.
*	**Model Persistence**: Uses joblib for efficient saving and loading of trained models.
*	**Data Visualization**: Provides comprehensive insights through correlation and performance plots. 

## 🛠 Dependencies:<br>
Install the required Python packages before running the project<br>
pip install pandas numpy matplotlib scikit-learn joblib

### 📊 DataSet information:<br>
The model analyzes the following features to determine classificat
*	Gender
*	Age Range,
*	Head Size(cm^3),
*	Brain Weight(grams)<br>

---

### ⚙️ Workflow:

#### 1.	Data Preparation: <br>
*	Loads MarvellousHeadBrain.csv into a Pandas DataFrame.
*	Converts all categorical or object-based columns into numeric values for processing. 

#### 2. Pipeline Construction:<br>
*	**Step 1(Scaling)** : Standard scalar to scale all the features<br>
*	**Step 2(Classifier)** : Used Decision Tree for result prediction
  
#### 3.	Model Training and Evaluation:<br>
*	**Metrics**: Evaluated via Accuracy Score, Confusion Matrix, and a full Classification Report.
*	**Feature Importance Plot**: Generates a plot to identify the most influential predictors. 
 

#### 4.	Model Saving and Loading:<br>
*	Saves the trained pipeline as a _.joblib_ file to bypass retraining in production environments.
---  
#### 🏃Running the Project:<br>
*	**The project reads data directly via**:
  
	import pandas as pd
	df = pd.read_csv("MarvellousHeadBrain.csv")

*	**Train and evaluate model**<br>
	To train the model from scratch and view performance metrics:<br>
	
	 python3 HeadBrainClassification.py --train<br>
*	**Test pretarined model**<br>
  	To load the saved model and run predictions on test data:
 	
	python3 HeadBrainClassification.py --test<br>  

---

#### 📂 Expected Outputs saved:

**Text Reports**
*	**PredictedResults.txt**: Output of the model's predictions on test data.
*	**classification_report.txt**: Detailed precision, recall, and F1-score breakdown. 


**Visualisations**<br>
*	**ConfusionMatrix.png**: Visual representation of true vs. predicted classes.
*	**CoRelationMatrix.png**: Heatmap showing relationships between features. 

**Model Storage**: <br>
Model saved as follows<br>
  -Decision Tree Classifier.joblib<br>

---	
 **✍️Author**

 Vaishali M. Jorwekar<br>
 Date	:10 Oct 2025 
  
  
  
