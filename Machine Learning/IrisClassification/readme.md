## ☘️ Iris Classification
## 📌 Overview:
This project implements classic Machine Learning classification on the Iris Dataset using Decision Tree and K-Nearest Neighbors (KNN). It adheres to industrial best practices by automating workflows and ensuring model reproducibility.

### Industrial Best Practices Followed:
*	**Automated Preprocessing**: Utilizes Scikit-Learn Pipeline for seamless data flow.
*	**Feature Scaling**: Implements StandardScaler to normalize dataset values.
*	**Model Persistence**: Saves and loads trained models using joblib to avoid redundant retraining.
*	**Data Visualization**: Generates insightful plots for feature analysis and model evaluation.
         			
---            
### 🛠  Dependencies:
Install the required Python packages before running the project
	_pip install pandas numpy matplotlib scikit-learn joblib_

### 📊 DataSet information:
The project utilizes Iris.csv consisting of the following variables:
<table>
  <thead>
    <tr>
      <th align="left">Type</th>
      <th align="left">Columns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Features</b></td>
      <td>sepal.length, sepal.width, petal.length, petal.width</td>
    </tr>
    <tr>
      <td><b>Target (Variety)</b></td>
      <td>setosa, versicolor, virginica</td>
    </tr>
  </tbody>
</table>

---

### ⚙️ Workflow:<br>
**1.	Data Preparation**:Convert Iris.csv into a Pandas DataFrame and ensure all feature values are numeric.<br>
**2.	Train-Test-Split**: Split the dataset into 80% Training and 20% Testing sets.<br>
**3.	Pipeline Construction**:<br>
*	**Step 1** : Standard scalar to scale all the features
*	**Step 2** : KNN and Decision Tree for result prediction<br>

**4.	Training & Evaluation**:<br>
*	Evaluate models using Accuracy, Confusion Matrix, and Classification Reports.<br>

**5.	Model Saving and Loading**:<br>
*	Save models via joblib for future inference.

---		
### 🚀 Running the Project:
*	**Train and evaluate model**:
Run the following command to train the pipelines and generate evaluation metrics:
	python3 IrisClassification.py --train
*	**Testing the model**:
  	python3 IrisClassification.py --test 
---    
### 📁 Expected Outputs & Storage:<br>
#### 📊 Visualizations & Reports
*	**Predicted Results**: PredictedResults.txt
*	**Confusion Matrix**: ConfusionMatrix.png
*	**Classification Report**: classification_report.txt
*	**Feature Plot**: featureWisePlot.png
  
#### 💿 Model Artifacts
Trained pipelines are saved in the artifact_Iris/ directory:
*	artifact_Iris/Decision Tree Classifier.joblib
*	artifact_Iris/KNN.joblib

---  
#### ✍️Author:<br>
Vaishali M. Jorwekar<br>
Date	: 30 Oct 2025

    

  
  

  
           


