# 🧠Brain Size vs. 😎Intelligence Regression Analysis

# 📌 Overview 
This project implements a machine learning regression model to analyze the correlation between head size and brain weight. It demonstrates industrial best practices, including automated preprocessing pipelines, feature scaling, and model persistence.

## 🚀 Key Features
It Follows industrial best practices by
*	**Automated Preprocessing**: Utilizes Scikit-Learn Pipeline to streamline data workflows.
*	**Feature Scaling**: Implements StandardScaler to normalize dataset values.
*	**Predictive Modeling**: Uses Linear Regression for weight estimation.
*	**Model Persistence**: Saves and loads trained models using joblib for efficient deployment.
*	**Comprehensive Visualization**: Generates correlation matrices and regression plots.
---
## 🛠 Dependencies:<br>
Install the required Python packages before running the project<br>
pip install pandas numpy matplotlib scikit-learn joblib

### 📊 DataSet information:<br>
#### Features and Target:<br>
The model processes the MarvellousHeadBrain.csv file with the following attributes:<br>
**Feature				Description**<br>
Gender					Categorical identifier<br>
Age Range				Categorical classification<br>
Head Size (cm³)			Independent variable (Predictor)<br>
Brain Weight (grams)	Target variable (Goal)<br>

---

### 🔄 Workflow:

#### 1.	Data Preparation: <br>
•  Loading CSV via pandas and converting data to numeric formats.

#### 2.	Pipeline Construction:<br>
*	**Step 1** : Standard scalar to scale all the features<br>
*	**Step 2** : Used Linear Regression for result prediction
  
#### 3.	Model Training and Evaluation:<br>
 Calculating Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and \(R^{2}\) Score.

#### 4.	Storage: 
Exporting the model object for future use without retraining.

---
#### 💻  Running the Project:<br>
1.	**Train and evaluate model**<br>
	  python3 HeadBrainRegression.py --train<br>
2.	**Test pretarined model**<br>
	python3 HeadBrainRegression.py --test<br>  

#### 📁 Expected Outputs saved:

**Artifacts & Metrics** :
*	**PredictedBrainWeight.txt**: Text file containing model predictions.
*	**MeanSquaredError.txt**: Log of MSE, RMSE, and \(R^{2}\) metrics.
*	**Linear Regression.joblib**: The serialized trained model.
**Visualizations**
*	**CoRelationMatrix.png**: Heatmap illustrating feature relationships.
*	**HeadSizeVsBrainWtRegressionPlot.png**: Scatter plot with the fitted regression line.


---
 **✍️Author**

 Vaishali M. Jorwekar<br>
 Date	:8 Oct 2025 
  
  
  
