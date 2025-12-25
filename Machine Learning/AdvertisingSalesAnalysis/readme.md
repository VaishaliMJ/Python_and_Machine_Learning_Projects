
# Advertisement Sales Prediction

Built a Regression model to predict advertisement sales.

## 🚀 Key Features
This project follows industrial best practices for machine learning:
- **Preprocessing**: Uses `sklearn.pipeline.Pipeline` to bundle scaling and modeling.
- **Standards Scaling**: Implements `StandardScalar` to normalize dataset values.
- **Algorithm Used**: **Linear Regression** for predictive analysis.
- **Model Save And Reload**: Saves and loads trained models using `joblib`.
- **Visualization**: Provides insights via correlation matrices and regression plots.

## 🛠 Dependencies
Install the required Python libraries before running the project:

		pip install pandas numpy matplotlib scikit-learn joblib<br>
---

## 📊 Dataset Information<br>
	The model analyzes the relationship between advertising budgets and sales.
- **Features** : TV, radio, newspaper
- **Target**: sales
---
### ⚙️  Workflow
1. **Data Preparation**
	- Converts Advertising.csv into a Pandas DataFrame.
	- Cleans and prepares the dataset for processing.

2.	**Pipeline Construction**
  	-	Step 1 (Scaling): StandardScalar to normalize all features.
	-	Step 2 (Modeling): LinearRegression for prediction.

  
3.	**Model Training and Evaluation**
  -	**Metrics**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and \(R^{2}\) Score.
  -	**Visuals**: Generates a Correlation Matrix to identify feature importance.
  
4. **Model Saving and Loading**:<br>
*	Saves the trained model as Linear Regression.joblib.
*	Allows for future predictions without the need for retraining.<br>
---

### 🚀 Running the Project
**Train and Evaluate**<br>

	python3 AdvertisingSalesAnalysis.py --train<br>
	
**Test pre-trained model**<br>

	python3 AdvertisingSalesAnalysis.py --test<br> 


### 📁 Expected Outputs


<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>File Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Data</td>
      <td><code>PredictedSales.txt</code></td>
      <td>List of predicted sales values</td>
    </tr>
    <tr>
      <td>Metrics</td>
      <td><code>MeanSquaredError.txt</code></td>
      <td>Final error calculations</td>
    </tr>
    <tr>
      <td>Visuals</td>
      <td><code>CoRelationMatrix.png</code></td>
      <td>Heatmap of feature correlations</td>
    </tr>
    <tr>
      <td>Visuals</td>
      <td><code>SalesPredictionPlot.png</code></td>
      <td>Plot comparing Actual vs Predicted values</td>
    </tr>
    <tr>
      <td>Model</td>
      <td><code>Linear Regression.joblib</code></td>
      <td>The serialized trained model</td>
    </tr>
  </tbody>
</table>

    
 **✍️Author**

 **Vaishali M. Jorwekar**<br>
 **Date	:25 Nov 2025**
  
         

