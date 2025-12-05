# Overview 
Customer Segmentation(Unsupervised -K Means clustering)-Grouped retail customers into distinct clusters based on purchasing behaviour for targeted marketing
It Follows industrial best practices by

		
•	Scaling the dataset values using StandardScalar
		
•	Used K Means clustering algorithm
		
•	Saving and loding trained model using job lib
		
•	Provided data visualisation

## Dependencies:<br>
Install the required Python packages before running the project<br>
_pip install pandas numpy matplotlib scikit-learn joblib streamlit_

### DataSet information:<br>
#### Features:<br>
ID,Year_Birth,Education,Marital_Status,Income,Kidhome,Teenhome,<br>
Dt_Customer,Recency,MntWines,MntFruits,MntMeatProducts,MntFishProducts,<br>
MntSweetProducts,MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,<br>
NumStorePurchases,NumWebVisitsMonth,AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,<br>
AcceptedCmp1,AcceptedCmp2,Complain,Z_CostContact,Z_Revenue,Response<br>

### Workflow:

#### Data Preparation: <br>
• Convert customer_segmentation.csv file into data frame<br>
•	Clean data set
• Select important features for model training

####  Pre-Processing :<br>
  •	Step 1 : Standard scalar to scale all the features<br>
	•	Step 2 : Used K-Means for result prediction

#### Model Training and Evaluation:<br>
  •	Found Optimal value of 'k'<br>
  •	Plotted Co-Relation Matrix<br>
  • Applied PCA and Plot clusters for visualisation<br>
#### Model Saving and Loading:<br>
  •	Save the model with joblib <br>
	•	Load models for future predictions without retraining<br>

 #### Running the Project:<br>
  •  Load data set (only once)<br>
      --->  pandas.read_csv(file_path)<br>
  •  Train and evaluate model<br>
	    --->  python3 CustomerSegmentation.py --train<br>
  •  Testing of the testModule<br>
      --->  streamlit run testModule.py<br>

 #### Expected Outputs saved:
 **Visualisations**<br>

  Co-Relation.png.png<br>

  ClusterVsInertiaPlot.png<br>

  DataAnalysis.png<br>

  IncomeWiseAnalysis.png<br>

  ClusterSummary.txt<br>

  SpendingAnalysis.png<br>

  **Model Storage**: Model saved as follows

  -CustomerSegmentation.joblib<br>
    
 **Author**

 Vaishali M. Jorwekar<br>
 Date	:28 Nov 2025 
  

  




