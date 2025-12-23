## 🚢 Titanic Passenger Survival Prediction
### 🚀 Overview: 
This project builds a classification system to predict passenger survival using "Logistic Regression" and "Decision Tree" algorithms. 
It adheres to industrial best practices by:
*	**Automating Preprocessing**: Using Scikit-Learn Pipeline.
*	**Feature Scaling**: Implementing StandardScaler for better model convergence.
*	**Model Persistence**: Saving and loading models using joblib to avoid retraining.
*	**Data Visualization**: Providing insights through correlation matrices and importance plots.
---
### 🛠 Dependencies:
Install the required Python packages before running the project

	pip install pandas numpy matplotlib scikit-learn joblib

### 📊 DataSet information:
The model processes the following features and Target Variable:
*	**Features**-Passengerid,Age,Fare,Sex,sibsp,Parch,zero,Pclass,Embarked,
*	**target** - Survived.

---

### 🔄  Workflow:
1.	####  Data Preparation:
*	**Ingestion**: Converts TitanicDataset.csv into a Pandas DataFrame.
*	**Imputation**: Replaces 0 values in specific columns with the column mean().
*	**Encoding**: Converts categorical string values into numeric values for model compatibility.

2.	####  Train-Test-Split:
*	Split data set into 80% Training and 20% Testing set

3.	####  Pipeline Construction:
   The process is streamlined using a two-step pipeline:
	*	**Step 1** : Standard scalar to scale all the features
	*	**Step 2** : Used Logistic Regression and Decision Tree for result prediction

4.	####  Evaluation & Metrics:
	*	**Metrics**: Accuracy, Confusion Matrix, and Scikit-Learn Classification Report.
	*	**Visuals**: A Feature Importance Plot identifies the most influential variables in predicting survival.
5.	####  Model Saving and Loading:
    *	Save the all three models with joblib 
	*	Load models for future predictions without retraining 
---
#### 💻 Running the Project:
*	##### Train and evaluate model
			python3 TitanicPassengerSurvival.py --train
*	##### Test pre-trained model
			python3 TitanicPassengerSurvival.py --test  

---			
####  📁 Expected Outputs:

  <table>
  <thead>
    <tr>
      <th align="left">File</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>PredictedResults.txt</code></td>
      <td>Output of the latest test predictions</td>
    </tr>
    <tr>
      <td><code>ConfusionMatrix.png</code></td>
      <td>Heatmap of True Positives vs. False Positives</td>
    </tr>
    <tr>
      <td><code>classification_report.txt</code></td>
      <td>Precision, Recall, and F1-Score breakdown</td>
    </tr>
    <tr>
      <td><code>CoRelationMatrix.png</code></td>
      <td>Correlation heatmap of all features</td>
    </tr>
  </tbody>
</table>


####  💿Model Storage: Model saved as follows
*	**Logistic Regression** : LogisticRegression.joblib
*	**Decision Tree** : Decision Tree Classifier.joblib

####  ✍️Author:
	 Vaishali M. Jorwekar
	 Date	:28 Sep 2025
