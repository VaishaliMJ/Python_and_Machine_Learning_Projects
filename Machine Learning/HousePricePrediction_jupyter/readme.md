# Overview 
Built Regression models to predict House Price using skleran dataset "california housing".Used Jupyter notebook

It Follows industrial best practices by

•	Automating preprocessing with Pipeline
		
•	Scaling the dataset values using StandardScalar
		
•	Used Linear Regression algorithm
		
•	Saving and loding trained model using pickle
		
•	Provided data visualisation

## Dependencies:<br>
Install the required Python packages before running the project<br>
pip install pandas numpy matplotlib scikit-learn pickle 

### DataSet information:<br>
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
  ### Workflow:

#### Data Preparation: <br>
• Convert "California Housing dataset" into data frame<br>
•	Convert all column values to numeric values

#### Pipeline Construction:<br>
  •	Step 1 : Standard scalar to scale all the features<br>
	•	Step 2 : Used Linear Regression for result prediction
  
#### Model Training and Evaluation:<br>
  •	Metrics: Mean squared error,Root mean squared error,R^2 <br>
  •	Co-Relation Matrix

#### Model Saving and Loading:<br>
  •	Save the model with pickle <br>
	•	Load models for future predictions without retraining<br>

#### Running the Project:<br>
  •	Load data set (only once)<br>
  •	Train and evaluate model<br>
  •	Test pretrained model<br>
  • Test App using Streamlit as well

#### Expected Outputs saved:

**Visualisations**<br>

  Co Realtion Matrix : CoRelationMatrix.png


**Model Storage**: Model saved as follows

  -lr_model.pkl<br>
    
 **Author**

 Vaishali M. Jorwekar<br>
 Date	:5 Nov 2025 
  

