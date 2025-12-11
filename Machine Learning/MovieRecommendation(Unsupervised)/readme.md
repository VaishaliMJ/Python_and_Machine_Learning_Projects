# Overview 
Built a recommendation engine using clustering & similarity measures<br>

This is content based recommendation system.

•  Used TfidfVectorizer for converting the text data into numerical values
		
•  Used K Means clustering algorithm
		
•  Saving and loding trained model using job lib
		
•  Provided data visualisation

•  Used Streamlit for testing the movie recommendation

## Dependencies:<br>
Install the required Python packages before running the project<br>
_pip install pandas numpy matplotlib scikit-learn joblib streamlit_<br>
_update config.json_ for API Key

### DataSet information:<br>
Used "tmdb_5000_movies.csv" and "tmdb_5000_credits.csv" movies dataset for building system

### Workflow:

#### Data Preparation: <br>
•  Convert tmdb_5000_movies.csv and tmdb_5000_credits.csv file into data frame<br>
•  Combine both datasets<br>
•  Select important fetaures only from both the datasets<br>
•	 Clean data set,by removing spaces,Stem to reduce word count<br>
	 
####  Pre-Processing :<br>
  •	Step 1 : Used TfidfVectorizer for convertion of text data<br>
	•	Step 2 : Used K-Means for result prediction

#### Model Training and Evaluation:<br>
  •	Found Optimal value of 'k'<br>
  •	Plotted Co-Relation Matrix<br>
  • Applied PCA and Plot clusters for visualisation<br>
  • Used cosine similarity for finding similar movies<br>
  
#### Model Saving and Loading:<br>
  •	Save the model with joblib <br>
	•	Load models for future predictions without retraining<br>
#### Running the Project:<br>
  •  Load data set (only once)<br>
      --->  pandas.read_csv(file_path)<br>
  •  Train and evaluate model<br>
	    --->  python3 MovieRecommondation.py --train<br>
  •  Testing of the testModule<br>
      --->  streamlit run MovieApp.py<br>
#### Expected Outputs :
 **Visualisations**<br>

  Co-Relation.png<br>

  ClusterVsInertiaPlot.png<br>
  

**Author**

 Vaishali M. Jorwekar<br>
 Date	:28 Nov 2025 
  


