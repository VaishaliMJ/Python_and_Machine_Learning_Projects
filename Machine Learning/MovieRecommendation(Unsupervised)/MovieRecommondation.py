"""---------------------------------------------------------------------------------------------------------------
                Movie Recommendation System(Unsupervised -K Means)
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Built a recommendation engine using clustering & similarity measures
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import os,ast,joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
ps=PorterStemmer()
from sklearn.metrics.pairwise import cosine_similarity



#####################################################################################################
#   Constants and file names
#####################################################################################################
BORDER="-"*65
MOVIES_DATA="tmdb_5000_movies.csv"
CREDITS_DATA="tmdb_5000_credits.csv"
ARTIFACT_DIR="artifact_MovieRecommondation"
MODEL_NAME="MovieRecommonder"
EXTRACTED_DATA_FRAME="ExtractedDataFrame"
RANDOM_STATE=42
###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
############################################################################################
def readCSVFile(csvFileName)->pd.DataFrame:
    dFrame=pd.read_csv(csvFileName)
    print(BORDER)
    print(f"Data loaded successfully from file '{csvFileName}'")
    print(BORDER)
    print(f"File Data:\n{BORDER}\n{dFrame.head()}")
    print(f"Data Set Shape:{dFrame.shape}")
    print(f"Columns in data set:{dFrame.columns}")
   
    print(BORDER)
    return dFrame    
###########################################################################################
#   Function        :   preProcessData
#   Input Params    :   None
#   Output Params   :   df(Data Frame)
#   Description     :   Load CSV data and pre-process data
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
###########################################################################################
def preProcessData():  
    #   Load data Movies from CSV file
    dfMovie=readCSVFile(MOVIES_DATA)  
    #   Load credits data from CSV file 
    dfCredits=readCSVFile(CREDITS_DATA)

    dfMovie=dfMovie.merge(dfCredits,on="title")
    print(f"Movies Data frame:{dfMovie.columns}")

    #   clean data set
    dfMovie=cleanDataSet(dfMovie)

    #   format Data Set
    dfMovie=formatDataSet(dfMovie)

    #   Update data frame
    dfMovieNew=dfMovie[['movie_id','title','tags']]
    dfMovieNew['tags']=dfMovieNew['tags'].apply(lambda x: " ".join(x)).apply(lambda x : x.lower())
    print(dfMovieNew)
    return dfMovieNew
###########################################################################################
#   Function        :   formatDataSet
#   Input Params    :   df(Data Frame)
#   Output Params   :   df(Data Frame)
#   Description     :   Cleans the data set
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
###########################################################################################
def formatDataSet(dfMovie):
    #   Format 'genres' column 
    dfMovie['genres']=dfMovie['genres'].apply(formatDataCols)  
    dfMovie['keywords']=dfMovie['keywords'].apply(formatDataCols) 
    
    dfMovie['crew']=dfMovie['crew'].apply(fetchDirector)
    dfMovie['cast']=dfMovie['cast'].apply(formatCast)

    #outDir=os.path.join(ARTIFACT_DIR,EXTRACTED_DATA_FRAME)
    # Save the DataFrame to a CSV file
    #dfMovie.to_dict(outDir, index=False)
    movies_dict=dfMovie.to_dict()
    saveModel(movies_dict,EXTRACTED_DATA_FRAME)
    dfMovie['overview']=dfMovie['overview'].apply(lambda x:x.split())
    
    
    #   Remove extra spaces
    dfMovie['genres']=dfMovie['genres'].apply(lambda x : [i.replace (" ","") for i in x])  
    dfMovie['keywords']=dfMovie['keywords'].apply(lambda x : [i.replace (" ","") for i in x])  
    dfMovie['cast']=dfMovie['cast'].apply(lambda x : [i.replace (" ","") for i in x])  
    dfMovie['crew']=dfMovie['crew'].apply(lambda x : [i.replace (" ","") for i in x])  

    #   Concat data drame
    dfMovie['tags']=dfMovie['overview']+dfMovie['crew']+\
                                    dfMovie['cast']+dfMovie['keywords']+dfMovie['genres']
    print(f"Updated Data frame:{dfMovie.head()}") 
    return dfMovie

###########################################################################################
#   Function        :   fetchDirector
#   Input Params    :   Data Frame: column
#   Output Params   :   Formatted Data Frame column
#   Description     :   Fetch Director
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
###########################################################################################
def fetchDirector(objList):
    featureList=[]
    
    for obj in ast.literal_eval(objList):
        if(obj['job']=='Director'):
             featureList.append(obj['name'])
             break     
    return featureList 

###########################################################################################
#   Function        :   formatCast
#   Input Params    :   Data Frame: column
#   Output Params   :   formatted Data Frame column
#   Description     :   format Data Frame column
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
###########################################################################################
def formatCast(obj):
    featureList=[]
    count=0
    for k in ast.literal_eval(obj):
        if(count != 3):
             featureList.append(k['name'])
             count+=1
        else:
            break      
    return featureList 

###########################################################################################
#   Function        :   formatGenres
#   Input Params    :   Data Frame: column
#   Output Params   :   formatted Data Frame column
#   Description     :   format Data Frame column
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
###########################################################################################
def formatDataCols(obj):
    featureList=[]
    for k in ast.literal_eval(obj):
        featureList.append(k['name'])
    return featureList    
##############################################################################################
#   Function        :   cleanDataSet
#   Input Params    :   df(Data Frame)
#   Output Params   :   df(Data Frame)
#   Description     :   Cleans the data set
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
###########################################################################################
def cleanDataSet(dfMovie):  
    dfMovie=dfMovie[["movie_id","genres","keywords","title","overview","cast","crew"]]
   
    print(f"Data Frame Info:{dfMovie.head()}")
    #   Null value coulmn drop
    print(f"Null value coulmns:\n{BORDER}\n{dfMovie.isnull().sum()}\n{BORDER}")
    dfMovie=dfMovie.dropna()
    #   Check for duplicate values
    print(f"Duplicates:{dfMovie.duplicated().sum()}\n{BORDER}")
    return dfMovie

###########################################################################################
#   Function        :   stemWords
#   Input Params    :   words
#   Output Params   :   None
#   Description     :   Stemming words
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
############################################################################################
def stemWords(text):
    stemWords=[]
    for word in text.split():
        stemWords.append(ps.stem(word))
    return " ".join(stemWords)    
###########################################################################################
#   Function        :   findKValue
#   Input Params    :   scaledFeatures
#   Output Params   :   None
#   Description     :   Find suitable 'k' value
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
############################################################################################
def findKValue(scaledFeatures):
    wcss=[]
    for k in range(1,21):
        kmeans=KMeans(n_clusters=k)
        kmeans.fit(scaledFeatures)
        print(f"K:{k} {kmeans.inertia_}")   #WCSS
        wcss.append(kmeans.inertia_)

    plt.plot(range(1,21),wcss,marker="o")
    plt.title("Elbow method for optimal K")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia(WCSS)")

    ensure_dir(ARTIFACT_DIR)
    plt.savefig(os.path.join(ARTIFACT_DIR,"ClusterVsInertiaPlot.png"))
    plt.close() 
###########################################################################################
#   Function        :   trainModel
#   Input Params    :   df,tfidf_matrix,nClusters
#   Output Params   :   None
#   Description     :   Train model using Kmeans
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
############################################################################################
def trainModel(df,tfidf_matrix,nClusters):  
    
    kmeans=KMeans(n_clusters=nClusters,random_state=RANDOM_STATE)
    y_KMeans=kmeans.fit_predict(tfidf_matrix)
    df["Cluster"]=y_KMeans  
    print(df.head(20))


    # Reduce dimensions
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(tfidf_matrix.toarray()) # convert sparse matrix to dense

    # Create scatter plot
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=y_KMeans, cmap='viridis', s=50, alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Movie Clusters Visualization (PCA)')
    ensure_dir(ARTIFACT_DIR)
    plt.savefig(os.path.join(ARTIFACT_DIR,"ClusterVisualisation.png"))
    plt.close() 
    return kmeans


###########################################################################################
#   Function        :   train_and_evaluate
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Training of model
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
############################################################################################
def train_and_evaluate():
    #   Pre-Process Data
    dFrame=preProcessData() 
    #   Stemming 
    dFrame['tags']=dFrame['tags'].apply(stemWords)
    
    #   Apply TF-IDF vectorizer
    tfv=TfidfVectorizer(stop_words="english",max_features=5000)
    tfidf_matrix=tfv.fit_transform(dFrame['tags'])
    
    print(dFrame.columns)

    
    #   K Means Algorithm
    findKValue(tfidf_matrix)
    #   Train Model
    kmeans=trainModel(dFrame,tfidf_matrix,nClusters=7)
    # Calculate cosine similarity matrix using the TF-IDF matrix
    cosine_sim = cosine_similarity(tfidf_matrix)

    #   Save data frame,tfidf_matrix and cosine similarity
    saveModelsAndMatrix(dFrame,kmeans,cosine_sim,tfidf_matrix)
    
#####################################################################################################
#   Function name    :  saveModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
#####################################################################################################
def saveModel(model,modelName):
    ensure_dir(ARTIFACT_DIR)
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)  
#####################################################################################################
#   Function name    :  saveModelsAndMatrix
#   Input Params     :  movies_df,knnModel,cosine_sim,tfidf_matrix
#   Output           :  -
#   Description      :  Save the trained model and Matrices
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
#####################################################################################################
def saveModelsAndMatrix(movies_df,knnModel,cosine_sim,tfidf_matrix):
    ensure_dir(ARTIFACT_DIR)
    #   Dump trained model
    saveModel(knnModel,MODEL_NAME)
    #   Save movies data frame

    movies_dict=movies_df.to_dict()
    saveModel(movies_dict,"movies_dict")

    #   Save Matrices
    saveModel(tfidf_matrix,"tfidf_matrix")

    saveModel(cosine_sim,'cosine_sim_matrix')
#####################################################################################################
#   Function Name    :  movieRecommondation 
#   Description      :  Manages calls to training and testing functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
#####################################################################################################
def movieRecommondation():
    ensure_dir(ARTIFACT_DIR)
    train_and_evaluate()
#########################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
#########################################################################################################
def main():
    movieRecommondation()

#########################################################################################################
#   Starter
#########################################################################################################
if __name__=="__main__":
    main()