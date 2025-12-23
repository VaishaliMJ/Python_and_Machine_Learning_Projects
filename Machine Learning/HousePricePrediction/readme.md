# 🏠 House Price Prediction Project
This project implements an industrial-grade machine learning workflow to predict residential property prices in India. It leverages Scikit-Learn's Pipeline architecture to ensure reproducible preprocessing and model deployment
# 📌 Overview 
## 🛠️ Core Components
Built Regression models to predict House Price

*	**Preprocessing**: Automated feature scaling using StandardScaler.
*	**Algorithm**: Multivariate Linear Regression.
*	**Serialization**: Model persistence using joblib for zero-retraining inference.
*	**Interface**: Interactive web application powered by Streamlit. 

## 🚀 Dependencies:<br>
Install the required Python packages before running the project<br>
pip install pandas numpy matplotlib scikit-learn joblib streamlit

### 📊 Dataset Schema:<br>
The model processes 22 features, including geographical coordinates, structural attributes, and proximity to infrastructure (schools/airports).
#### Features and Target:<br>
<table>
  <thead>
    <tr>
      <th align="left">Category</th>
      <th align="left">Key Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Physical</b></td>
      <td>Bedrooms, Bathrooms, Living Area, Basement Area, Number of Floors</td>
    </tr>
    <tr>
      <td><b>Condition</b></td>
      <td>House Grade, Condition, Built Year, Renovation Year</td>
    </tr>
    <tr>
	<td><b>Location</b></td>
      <td>Postal Code, Latitude, Longitude, Distance from Airport</td>
    </tr>
    <tr>
      <td><b>Target</b></td>
      <td>Price (Numeric)</td>
    </tr>
  	</tbody>
	</table>

---
### 🚀 Workflow:

#### 1. Data Preparation: <br>
*	Loads "House Price India.csv" into a Pandas DataFrame.
*	Forces numeric conversion for all columns to ensure mathematical compatibility.

#### 2.	Pipeline Construction:<br>
*	**Step 1** : StandardScaler to normalize feature scales.<br>
*	**Step 2** : Used Linear Regression for price prediction
  
#### 3.	Model Training and Evaluation:<br>
The model is evaluated using standard regression metrics:
*	R² Score: Accuracy of fit.
*	RMSE: Root Mean Squared Error.
*	Correlation Matrix: Heatmap visual for feature dependency.

#### 4.	Model Saving and Loading:<br>
*	**Export**: Model is serialized into Linear_Regression.joblib.
*	**Import**: Allows instant inference on new data without retraining.
---  
  
#### 🏃 Running the Project:<br>
*	**Train and evaluate model**<br>
	 python3 HousePricePrediction.py --train<br>
*	**Test pretrained model**<br>
	 python3 HousePricePrediction.py --test<br> 
*	**Test App using Streamlit as well(Web Application)**
     streamlit run PricePrediction.py

        

#### 📂 Expected Outputs:

**Visualizations**<br>

*	**Correlation Matrix**: CoRelationMatrix.png (Displays feature relationships).
*	**Regression Plot**: HousePricesRegressionPlot.png (Comparison of Actual vs. Predicted values).
**Model Storage**: Model saved as follows

**Model Files**<br>
Linear Regression: Linear_Regression.joblib
    
 **✍️Author**

 Vaishali M. Jorwekar<br>
 Date	:5 Nov 2025 
  
