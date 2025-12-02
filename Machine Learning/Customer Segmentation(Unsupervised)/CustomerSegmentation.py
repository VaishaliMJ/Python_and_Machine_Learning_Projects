"""---------------------------------------------------------------------------------------------------------------
                       Customer Segmentation(Unsupervised -K Means clustering)   
                        Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Customer Segmentation(Unsupervised -K Means clustering)-
                Grouped retail customers into distinct clusters based on purchasing behaviour for targeted marketing
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import os,argparse
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

#####################################################################################################
#   Constants and file names
#####################################################################################################
BORDER="-"*65
DATASET_FILENAME="customer_segmentation.csv"
ARTIFACT_DIR="artifact_CustomerSegmentation"
MODEL_NAME="CustomerSegmentation"
RANDOM_STATE=42
###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,interference baselines
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def parse_args():  
    p=argparse.ArgumentParser(description="Customer Segmentation based on Purchasing Behaviour")
    p.add_argument("--train",action="store_true",help="Training using KMeans algorithm")
    p.add_argument("--test",action="store_true",help="Testing of pre-trained model")
    p.add_argument("--samples",type=int,default=10,help="Number of samples for testing")
    return p.parse_args()
###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def readCSVFile(csvFileName=DATASET_FILENAME)->pd.DataFrame:
    dFrame=pd.read_csv(csvFileName)
    print(BORDER)
    print(f"Data loaded successfully from file '{csvFileName}'")
    print(BORDER)
    print(f"File Data:\n{BORDER}\n{dFrame.describe()}")
    print(f"Data Set Shape:{dFrame.shape}")
    print(f"Columns in data set:{dFrame.columns}")
   
    print(BORDER)
    return dFrame
###########################################################################################
#   Function        :   cleanDataSet
#   Input Params    :   df(Data Frame)
#   Output Params   :   df(Data Frame)
#   Description     :   Cleans the data set
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
###########################################################################################
def cleanDataSet(dFrame):    
    print(BORDER)
    print("Cleaning Data set....")
    print(BORDER)
    print(f"Null value check:\n{dFrame.isna().sum()}")
    print("Dimensions of data set before cleaning is :",dFrame.shape)
    print(f"Dropping 'na' value records:")
    dFrame.dropna(inplace=True)
    print("Dimensions of data set after removing 'na' values:",dFrame.shape)
    print(f"Data Set Info:\n{dFrame.info()}")
    dFrame["Dt_Customer"]=pd.to_datetime(dFrame["Dt_Customer"],dayfirst=True)
    print(f"Data Set Info:\n{dFrame.info()}")

    #   Find current year
    current_year = datetime.now().year
    dFrame['Age']=current_year-dFrame['Year_Birth']
    #   Number of children
    dFrame['Total_Children']=dFrame['Kidhome']+dFrame['Teenhome']
    #   Spending
    sepndingCols=['MntWines','MntFruits','MntMeatProducts','MntFishProducts',
                  'MntSweetProducts','MntGoldProds']
    dFrame['Spending']=dFrame[sepndingCols].sum(axis=1)
    #   Customer since
    dFrame['JoiningDate']=(datetime.now()-dFrame['Dt_Customer']).dt.days
    print(dFrame.head())

    print(dFrame['Marital_Status'].value_counts())
    return dFrame
###########################################################################################
#   Function        :   preProcessData
#   Input Params    :   None
#   Output Params   :   df(Data Frame)
#   Description     :   Load CSV data and pre-process data
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
###########################################################################################
def preProcessData():
    #   Load data from CSV file
    df=readCSVFile(csvFileName=DATASET_FILENAME)
    #   Clean data set
    df=cleanDataSet(df)
    #   Plot hsitogram
    plotDataAnalysis(df)
    #  plot Income wise details
    plotIncomeWiseDetails(df) 
    #   Display co-realation matrix
    displayCorrelationMatrix(df)
    return df
#####################################################################################################
#   Function Name   :   displayCorrelationMatrix
#   Description     :   Plot the Co-relation matrix
#   Input Params    :   Data frame   
#   Output          :   Co-relation Matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
#####################################################################################################
def displayCorrelationMatrix(df):
    ensure_dir(ARTIFACT_DIR)
    #   Co-realtion Matrix
    corr_col=df[["Income","Age","Spending","Recency","NumWebPurchases","NumStorePurchases"]].corr()
    plt.figure(figsize=(20,10))
    sns.heatmap(corr_col,annot=True,cmap="coolwarm")
    plt.title("Feature co-relation heatmap")
    plt.savefig(os.path.join(ARTIFACT_DIR,"Co-Relation.png"))
    plt.close()

    #   Average income by Education and Marital Status  
    pivot_income_df=df.pivot_table(values="Income",index="Education",columns="Marital_Status",aggfunc="mean")
    print(pivot_income_df)
    plt.figure(figsize=(20,10))
    sns.heatmap(pivot_income_df,annot=True,fmt="0.2f",cmap="YlGnBu")
    plt.title("Average income by Education and Marital Status")
    plt.savefig(os.path.join(ARTIFACT_DIR,"AverageIncome.png"))
    plt.close()

    fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(20,18)) 
    #   Average Spending by Education
    educationDF=df.groupby("Education")["Spending"].mean().sort_values(ascending=False)
    educationDF.plot(kind="bar",color="darkblue",ax=axes[0][0])
    axes[0][0].set_title("Average Spending by Education")
    axes[0][0].tick_params(axis='x', labelrotation=25)  

    #   Campaign accepted
    df["AcceptedAnyCamp"]=df[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"]].sum(axis=1)
    df["AcceptedAnyCamp"]=df["AcceptedAnyCamp"].apply(lambda x:1 if x>0  else 0)
    acceptedCampDF=df.groupby("Marital_Status")["AcceptedAnyCamp"].mean().sort_values(ascending=False)
    acceptedCampDF.plot(kind="bar",color="skyblue",ax=axes[0][1])
    axes[0][1].set_title("Campaign Acceptance by Marital Status")
    axes[0][1].tick_params(axis='x', labelrotation=25)  


    #   Age wise income
    bins=[18,30,40,50,60,70,90]
    labels=["18-29","30-39","40-49","50-59","60-69","70+"]
    df["AgeGroup"]=pd.cut(df["Age"],bins=bins,labels=labels)
    ageDF=df.groupby("AgeGroup",observed=False)["Income"].mean()
    ageDF.plot(kind="bar",color="orange",ax=axes[1][0])
    axes[1][0].set_title("Average Income by Age Group")


    plt.savefig(os.path.join(ARTIFACT_DIR,"SpendingAnalysis.png"))
    plt.close()


###########################################################################################
#   Function        :   plotIncomeWiseDetails
#   Input Params    :   data frame
#   Output Params   :   None
#   Description     :   Plots data frame details
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
###########################################################################################
def plotIncomeWiseDetails(dFrame): 
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,5)) 
    #   Education VS Income Plot
    
    sns.boxplot(x="Education",y="Income",data=dFrame,color='lightgreen',ax=axes[0])
    axes[0].set_xlabel("Education")
    axes[0].set_ylabel("Income")
    axes[0].set_title("Income Vs Education Plot")



    sns.boxplot(x="Marital_Status",y="Spending",data=dFrame,color='blue',ax=axes[1])
    axes[1].set_xlabel("Marital Status")
    axes[1].set_ylabel("Spending")
    axes[1].set_title("Marital_Status Vs Spending")

    plt.tight_layout()

    plt.savefig(os.path.join(ARTIFACT_DIR,"IncomeWiseAnalysis.png"))
    plt.close()   


###########################################################################################
#   Function        :   plotDataAnalysis
#   Input Params    :   data frame
#   Output Params   :   None
#   Description     :   Plots data analysis
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
###########################################################################################
def plotDataAnalysis(dFrame):    
    fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(20,8)) 

    #   Age distribution
    sns.histplot(dFrame['Age'],bins=30,ax=ax[0,0])
    ax[0,0].set_title("Age Distribution")

    #   Income Distribution
    sns.histplot(dFrame['Income'],bins=30,ax=ax[0,1])
    ax[0,1].set_title("Income Distribution")

    #   Total Spending Distribution
    sns.histplot(dFrame['Spending'],bins=30,ax=ax[1,0])
    ax[1,0].set_title("Spending Distribution")
    

    #   Education
    sns.histplot(dFrame['Education'],bins=10,ax=ax[1,1])
    ax[1,1].set_title("Education Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR,"DataAnalysis.png"))
    plt.close()   

###########################################################################################
#   Function        :   extractFeatures
#   Input Params    :   Data Frame
#   Output Params   :   None
#   Description     :   Extract features from data frame
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def extractFeatures(df):
    featureColNames=["Age","Spending","Income","NumWebPurchases","NumStorePurchases","NumWebVisitsMonth","Recency"]
    features=df[featureColNames].copy()
    return features
###########################################################################################
#   Function        :   save_scalar
#   Input Params    :   scalar
#   Output Params   :   None
#   Description     :   Save scalar
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def save_scalar(scalar):
    ensure_dir(ARTIFACT_DIR)
    path=os.path.join(ARTIFACT_DIR,"scalar.joblib")
    joblib.dump(scalar,path)
###########################################################################################
#   Function        :   scaleFeatures
#   Input Params    :   features
#   Output Params   :   None
#   Description     :   Scale the features
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def scaleFeatures(features):
    scaler=StandardScaler()
    scaled_features=scaler.fit_transform(features)
    #   Save scalar model
    save_scalar(scaler)
    return scaled_features
###########################################################################################
#   Function        :   findKValue
#   Input Params    :   scaledFeatures
#   Output Params   :   None
#   Description     :   Find suitable 'k' value
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def findKValue(scaledFeatures):
    wcss=[]
    for k in range(1,11):
        kmeans=KMeans(n_clusters=k)
        kmeans.fit(scaledFeatures)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1,11),wcss,marker="o")
    plt.title("Elbow method for optimal K")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia(WCSS)")

    ensure_dir(ARTIFACT_DIR)
    plt.savefig(os.path.join(ARTIFACT_DIR,"ClusterVsInertiaPlot.png"))
    plt.close() 
###########################################################################################
#   Function        :   trainModel
#   Input Params    :   df,scaledFeatures,nClusters
#   Output Params   :   None
#   Description     :   Train model using Kmeans
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def trainModel(df,X,nClusters):  
    kmeans=KMeans(n_clusters=nClusters,random_state=RANDOM_STATE)
    y_KMeans=kmeans.fit_predict(X)
    df["Cluster"]=y_KMeans

    plt.figure(figsize=(10,8))
    plt.scatter(X[y_KMeans==0,0],X[y_KMeans==0,1],s=80,c="red",label="Cluster 0: Low Income,Low Spending")
    plt.scatter(X[y_KMeans==1,0],X[y_KMeans==1,1],s=80,c="cyan",label="Cluster 1: High Income,Average Spending")
    plt.scatter(X[y_KMeans==2,0],X[y_KMeans==2,1],s=80,c="blue",label="Cluster 2:Low Income,Average Spending")
    plt.scatter(X[y_KMeans==3,0],X[y_KMeans==3,1],s=80,c="yellow",label="Cluster 3:Average Income,Average Spending")
    plt.scatter(X[y_KMeans==4,0],X[y_KMeans==4,1],s=80,c="darkblue",label="Cluster 4:High Income,High Spending")
    plt.scatter(X[y_KMeans==5,0],X[y_KMeans==5,1],s=80,c="pink",label="Cluser 5:Average Income,Low Spending")

    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='black',label='centroid')
    plt.xlabel("Annual Income")
    plt.ylabel("Spending")
    plt.title("Cluster Analysis")
    plt.legend()
    plt.legend()
    plt.savefig(os.path.join(ARTIFACT_DIR,"Cluster_Analysis.png"))
    plt.close()
    print(df.head())
    return df,kmeans
###########################################################################################
#   Function        :   plotClusters
#   Input Params    :   df,scaled_features
#   Output Params   :   None
#   Description     :   Plot clusters
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def plotClusters(df,X):
    features=["Age","Spending","Income","NumWebPurchases","NumStorePurchases","NumWebVisitsMonth","Recency"]
    clusterGroups=df.groupby("Cluster")[features].mean()
    clusterSummary=f"\n Cluster Summary:\n{BORDER}{BORDER}\n{clusterGroups}\n{BORDER}{BORDER}"
    print(clusterSummary)
    with open(os.path.join(ARTIFACT_DIR,"ClusterSummary.txt"),"w") as f:
        f.write(clusterSummary)


    pca=PCA(n_components=2)
    pca_cols=pca.fit_transform(X)
    df["PCA1"]=pca_cols[:,0]
    df["PCA2"]=pca_cols[:,1]

    fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(15,10))
    sns.scatterplot(x="PCA1",y="PCA2",hue="Cluster",data=df,palette="Set2",ax=axes[0])
    axes[0].set_title("Customer Segmentation with PCA")
    #plt.title("Customer Segmentation with PCA")
    

    sns.scatterplot(x="Income",y="Spending",data=df,hue="Cluster",palette="Set1",ax=axes[1])
    #plt.title("Cluster's profile based on Income and Spending")
    axes[1].set_title("Cluster's profile based on Income and Spending")

    plt.legend()
    plt.savefig(os.path.join(ARTIFACT_DIR,"CustomerSegmentation.png"))
    plt.close()
#####################################################################################################
#   Function name    :  saveTrainedModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
#####################################################################################################
def saveTrainedModel(model):
    ensure_dir(ARTIFACT_DIR)
    path=os.path.join(ARTIFACT_DIR,MODEL_NAME+".joblib")
    joblib.dump(model,path)    
###########################################################################################
#   Function        :   train_and_evaluate
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Training of model
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def train_and_evaluate():
    #   Pre-Process Data
    dFrame=preProcessData()
    #   Extract features
    features=extractFeatures(dFrame)
    #   Scale features
    scaled_features=scaleFeatures(features)
    #   Find 'k' values
    findKValue(scaled_features)
    #   Train the model
    dFrame,model=trainModel(dFrame,scaled_features,nClusters=6)
    #   Plot cluster
    plotClusters(dFrame,scaled_features)
    #   Save model
    saveTrainedModel(model)


#####################################################################################################
#   Function Name    :  customerSegmentation 
#   Description      :  Manages calls to training and testing functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
#####################################################################################################
def customerSegmentation():
    ensure_dir(ARTIFACT_DIR)
    args=parse_args()
    did_anything=False
    if args.train:
        train_and_evaluate()
        did_anything=True   
    if not did_anything:
        print(
            "Nothing to do .Try one of the :\n"
            "python3 CustomerSegmentation.py --train\n"
        )    


#########################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
#########################################################################################################
def main():
    customerSegmentation()

#########################################################################################################
#   Starter
#########################################################################################################
if __name__=="__main__":
    main()