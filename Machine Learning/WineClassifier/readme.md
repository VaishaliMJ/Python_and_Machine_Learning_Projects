## 🍷Wine Type Classification 
This project implements machine learning models for the multi-class classification of wine types using K-Nearest Neighbors (KNN) and Random Forest. It follows industrial best practices by automating the workflow with scikit-learn Pipelines and ensuring model persistence.

---

### Overview 
Implmented ML models for Multi-class classification of Wine Type using KNN and Random Forest
### 🚀 Key Features
It Follows industrial best practices by
*	**Pipeline Automation**: Streamlined preprocessing and modeling.
*	**Feature Scaling**: Data normalization using StandardScaler.
*	**Model Persistence**: Training once and saving models via joblib.
*	**Comprehensive Visualization**: Automated generation of Confusion Matrices and Feature Importance plots.

---

### 🛠 Dependencies:<br>
Install the required Python packages before running the project<br>
pip install pandas numpy matplotlib scikit-learn joblib

### 📊 DataSet information:<br>
The dataset contains chemical analysis of wines grown in the same region but derived from three different cultivars.
*	**Features**: Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines, Proline.
*	**Target**: Class (Multi-class labels).
---

### ⚙️ Workflow:

1.	#### Data Preparation: <br>
*	Load WinePredictor.csv into a Pandas DataFrame.
*	Enforce numeric types across all columns and handle missing values.

2.	#### Pipeline Construction:<br>
*	**Step 1** : Standard scalar to scale all the features<br>
*	**Step 2** : Used KNN and Random Forest for result prediction
  
3.	#### Model Training and Evaluation:<br>
*	**Metrics**: Accuracy, Confusion Matrix, and Classification Report.
*	**Feature Importance**: Visualized to identify the most influential features.

4.	#### Model Save:<br>
*	Save the all two models with joblib <br>
*	Load models for future predictions without retraining<br>
---
### 🏃🏻‍♂️Running the Project:<br>
*	**Train and evaluate model**<br>
	 python3 WinePredictor.py --train<br>
*	**Test pretarined model**<br>
     python3 WinePredictor.py --test<br>  
---
#### 📂 Expected Outputs saved:

*	**Confusion Matrix**: ConfusionMatrix.png (One per model).
*	**Classification Report**: classification_report.txt.
*	**Feature Importance**: FeatureImportance.png (Random Forest analysis).
*	**Predictions**: PredictedResults.txt (Output from test mode).

##### Model Storage**: Model saved as follows
*	Random Forest:Random Forest Classifier.joblib<br>
*	KNN : KNN.joblib
---    
##### ✍️Author
 Vaishali M. Jorwekar<br>
 Date	:21 Oct 2025 
  
  
  
