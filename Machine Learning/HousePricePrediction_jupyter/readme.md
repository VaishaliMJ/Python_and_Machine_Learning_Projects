# 🏠 California House Price Prediction
A machine learning project that builds a regression model to predict housing prices using the Scikit-Learn California Housing dataset. This project implements industrial best practices, including automated preprocessing pipelines and model persistence.
## 🚀 Overview 
This project uses a Linear Regression model to estimate house prices based on geographic and demographic features. It focuses on clean code structure, data visualization, and deployment readiness. 
### 📌Key Features:
Built Regression models to predict House Price using skleran dataset "california housing".Used Jupyter notebook

It Follows industrial best practices by

•	Automating preprocessing with Pipeline
		
•	Scaling the dataset values using StandardScalar
		
•	Used Linear Regression algorithm
		
•	Saving and loding trained model using pickle
		
•	Provided data visualisation

### 🛠️Dependencies:<br>
Install the required Python packages before running the project<br>
pip install pandas numpy matplotlib scikit-learn pickle 

### 📊 DataSet information:<br>
#### Features and Target:<br>
  - MedInc        median income in block group
  - HouseAge      median house age in block group
  - AveRooms      average number of rooms per household
  - AveBedrms     average number of bedrooms per household
  - Population    block group population
  - AveOccup      average number of household members
  - Latitude      block group latitude
  - Longitude     block group longitude
  - Price         House Prices
---    
### ⚙️ Workflow:

#### 1.	Data Preparation: <br>
*	Load the dataset and convert it into a Pandas DataFrame.
*	Ensure all column values are numeric and handle any missing data

#### 2.	Pipeline Construction:<br>
The model utilizes a Scikit-Learn Pipeline to ensure that data scaling and regression are treated as a single atomic step: 
*	**Step 1** : Standard scalar to scale all the features<br>
*	**Step 2** : Used Linear Regression for result prediction
  
#### 3.Model Training and Evaluation:<br>
*	Mean Squared Error (MSE)
*	Root Mean Squared Error (RMSE)
*	R-squared (\(R^{2}\))

#### 4.	Model Saving and Loading:<br>
*	The trained pipeline is serialized into **lr_model.pkl**.
*	This allows for future predictions without the need to retrain the model. 
---
#### 🏃🏻‍♂️Running the Project:<br>
*	Open the Jupyter Notebook to train the model:
*	Run the cells to train, evaluate, and save lr_model.pkl. 
---

#### 📂 Expected Outputs saved:

**Visualisations**<br>
*	**CoRelationMatrix.png**: A heatmap showing the correlation between features and the target price


**Model Storage**:<br> 

*	**lr_model.pkl**: The serialized Scikit-Learn pipeline containing the scaler and the linear regression coefficients.

---   
 **✍️Author**

 Vaishali M. Jorwekar<br>
 Date	:5 Nov 2025 
  

