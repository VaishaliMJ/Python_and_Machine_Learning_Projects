# Customer Segmentation (Unsupervised K-Means)

# 📌 Overview 
This project groups retail customers into distinct clusters based on purchasing behavior to enable targeted marketing strategies. By utilizing **K-Means Clustering**, the model identifies patterns in demographics and spending habits to help businesses optimize their promotional efforts.

## Key Industrial Best Practices:
*	**Feature Scaling**: Implemented StandardScaler for uniform data distribution.
*	**Dimensionality Reduction**: Used PCA (Principal Component Analysis) for high-dimensional data visualization.
*	**Model Persistence**: Utilized joblib for saving and loading the trained model.
*	**Interactive UI**: Provided a Streamlit dashboard for real-time testing.
---
## ⚙️ Installation & Dependencies:<br>
Install the required Python packages before running the project<br>
_pip install pandas numpy matplotlib scikit-learn joblib streamlit_

### 📊 DataSet information:<br>
#### Features:<br>
ID,Year_Birth,Education,Marital_Status,Income,Kidhome,Teenhome,<br>
Dt_Customer,Recency,MntWines,MntFruits,MntMeatProducts,MntFishProducts,<br>
MntSweetProducts,MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,<br>
NumStorePurchases,NumWebVisitsMonth,AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,<br>
AcceptedCmp1,AcceptedCmp2,Complain,Z_CostContact,Z_Revenue,Response<br>

### 🚀 Workflow:

#### 1.	Data Preparation: <br>
*	Data cleaning and feature engineering from customer_segmentation.csv.
*	Scaling features to ensure the K-Means algorithm treats all attributes with equal importance.
####  2. Model Training & Evaluation
*	**Optimal 'k' Selection**: Determining the number of clusters using the Elbow Method.
*	**Correlation Analysis**: Plotting matrices to find relationships between spending and income.
*	**Visualization**: Applying PCA to project clusters into a 2D space for analysis.<br>

#### 3. Model Storage
  The model is serialized to CustomerSegmentation.joblib for deployment without retraining.

---
#### 💻 Running the Project:<br>
*	**Train the Model**
	To load the data, train the algorithm, and generate visualizations:<br>
	python3 CustomerSegmentation.py --train<br>
*	**Launch the Dashboard**
	To test specific customer data points through the Streamlit interface:<br>
    streamlit run testModule.py<br>

 #### 📊 Expected Outputs saved:
 The following insights are automatically saved to the project directory:
*	**Co-Relation.png**: Feature dependency heatmap.
*	**ClusterVsInertiaPlot.png**: The "Elbow" graph used to find optimal clusters.
*	**DataAnalysis.png / IncomeWiseAnalysis.png**: Demographic distributions.
*	**SpendingAnalysis.png**: Behavioral breakdown per cluster.
*	**ClusterSummary.txt**: A descriptive summary of each customer segment.


  #### Model File: 
  Model saved as follows

  *	CustomerSegmentation.joblib<br>
    
 **👩‍💻 Author**

 Vaishali M. Jorwekar<br>
 Date	:28 Nov 2025 
  

  




