"""---------------------------------------------------------------------------------------------------------------
            Sentiment Analysis Enhancement (Ensemble Bagging & Boosting)
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Sentiment Analysis Enhancement (Random Forest and Gradient Boosting)
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import os,joblib
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score,classification_report,ConfusionMatrixDisplay
 

#####################################################################################################
#   Constants and file names
#####################################################################################################
BORDER="-"*65
TWEETS_DATA="Tweets.csv"
ARTIFACT_DIR="artifact_sentimentAnalysis"
MODEL_NAME="sentimentAnalysis"
RANDOM_STATE=42
TEST_SIZE=0.2
NUM_ESTIMATORS=100
LEARNING_RATE=0.1
TARGET_COLUMN="sentiment"
"""Classification model list"""
CLF_MODELS={"Random Forest Classifier":RandomForestClassifier(n_estimators=NUM_ESTIMATORS,random_state=0),
            "Gradient Boosting Classifier":GradientBoostingClassifier(n_estimators=NUM_ESTIMATORS,
                                                                      learning_rate=LEARNING_RATE,random_state=0)}


###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
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
##############################################################################################
#   Function        :   cleanDataSet
#   Input Params    :   dFrame(Data Frame)
#   Output Params   :   dFrame(Data Frame)
#   Description     :   Cleans the data set
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
###########################################################################################
def cleanDataSet(dFrame):  
    dFrameUpdated=dFrame[["text","sentiment"]]
    print(f"Data Frame Info:{dFrame.head()}")
    #   Null value coulmn drop
    print(f"Null value coulmns:\n{BORDER}\n{dFrame.isnull().sum()}\n{BORDER}")
    dFrameUpdated=dFrameUpdated.dropna()
    #   Check for duplicate values
    print(f"Duplicates:{dFrameUpdated.duplicated().sum()}\n{BORDER}")
    return dFrameUpdated
###########################################################################################
#   Function        :   textDataProcessing
#   Input Params    :   dFrame
#   Output Params   :   df(Data Frame)
#   Description     :   pre-process text column 
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
###########################################################################################
def textDataProcessing(dFrame):  
     
    dFrame['cleanedText']=dFrame['text'].apply(processAndStemText)
        
    return dFrame
###########################################################################################
#   Function        :   processAndStemText
#   Input Params    :   text
#   Output Params   :   None
#   Description     :   pre-process text column
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
############################################################################################
def processAndStemText(text):
    text=text.lower()
    #   Alphabets only
    text = re.sub(r"[^a-zA-Z\s]", "", text)  
    #   remove https ,www,@ 
    text=re.sub(r'http\S+|www\S+',"",text)
    #   Remove extra white spaces
    text = re.sub(r"\s+", " ", text).strip() 
    
    stemWords=[]
    ps=PorterStemmer()
    for word in text.split():
        stemWords.append(ps.stem(word))
    return " ".join(stemWords)  
    
###########################################################################################
#   Function        :   stemWords
#   Input Params    :   words
#   Output Params   :   None
#   Description     :   Stemming words
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
############################################################################################
def stemWords(text):
    stemWords=[]
    ps=PorterStemmer()
    for word in text.split():
        stemWords.append(ps.stem(word))
    return " ".join(stemWords)        
###########################################################################################
#   Function        :   preProcessData
#   Input Params    :   None
#   Output Params   :   df(Data Frame)
#   Description     :   Load CSV data and pre-process data
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
###########################################################################################
def preProcessData():  
    #   Load data Movies from CSV file
    df=readCSVFile(TWEETS_DATA) 
    #   Remove 'na' and 'duplicates'
    dFrame=cleanDataSet(df)  
    #   Convert text into vectors
    dFrame=textDataProcessing(dFrame) 
    pd.set_option('display.max_columns', None)
    #   Plot sentiment distribution
    plotSentimentDistribution(dFrame)
    #Label enconding for all 'sentiment'
    #   Neutral : 1. negative :0, positive : 2
    labelEncoder = LabelEncoder()
    dFrame['sentiment']=labelEncoder.fit_transform(dFrame['sentiment'])
    
    print(f"{BORDER}\nData set after stemming words:\n{dFrame.tail()}")
    
    return dFrame 
#####################################################################################################
#   Function name    :  plotSentimentDistribution
#   Input Params     :  dFrame
#   Output           :  -
#   Description      :  Plot sentiment distribution
#   Author           :  Vaishali M Jorwekar
#   Date             :  19 Dec 2025
#####################################################################################################
def plotSentimentDistribution(dFrame):
    fig = plt.figure(figsize=(7,5))
    color = ['grey','green','red']
    dFrame['sentiment'].value_counts().plot(kind='bar',color = color)
    plt.title('Sentiment distribution')
    plt.ylabel('Count')
    plt.xlabel('Sentiment')
    plt.grid(False)
    plt.gcf().autofmt_xdate() 
    plt.savefig(os.path.join(ARTIFACT_DIR,"SentimentDistribution.png"))
    plt.close()
#####################################################################################################
#   Function name    :  saveModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
#####################################################################################################
def saveModel(model,modelName):
    ensure_dir(ARTIFACT_DIR)
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)  
###########################################################################################
#   Function        :   spliDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   Training and testing data set
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
############################################################################################
def spliDataSet(dFrame,tfidf_matrix):  
    target=dFrame[TARGET_COLUMN] 
    x_Train,x_Test,y_Train,y_Test=train_test_split(tfidf_matrix,
                                                   target,
                                                   test_size=TEST_SIZE,
                                                   random_state=RANDOM_STATE)
    
    return x_Train,x_Test,y_Train,y_Test   
#####################################################################################################
#   Function name    :  trainModel
#   Input Params     :  model,xTrain,yTrain data set
#   Output           :  Return pipeline object
#   Description      :  Train a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
#####################################################################################################
def trainModel(model,xTrain,yTrain):
    model.fit(xTrain,yTrain)
    return model   
###########################################################################################
#   Function        :   trainAndFindAccuracy
#   Input Params    :   df,tfidf_matrix,nClusters
#   Output Params   :   None
#   Description     :   Train model using Kmeans
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
############################################################################################
def trainAndFindAccuracy(dFrame,tfidf_matrix): 
    
    features= dFrame['cleanedText']
    target=dFrame[TARGET_COLUMN]
    x_Train,x_Test,y_Train,y_Test=spliDataSet(dFrame,tfidf_matrix,)
    algorithmDetails={"Algorithm Name":[],"TrainedModel":[],"Accuracy Score":[],
                      "Confusion Matrix":[],"Classification Report":[]}
    
   
    for clf_Name,clfModel in CLF_MODELS.items():
        #   Train model
        trainedModel=trainModel(clfModel,x_Train,y_Train)
        
        algorithmDetails["Algorithm Name"].append(clf_Name)
        algorithmDetails["TrainedModel"].append(trainedModel)
        #   Predict result
        # Model accuracy calculations
        testModelAndAccuracyCalculation(trainedModel,x_Test,y_Test,algorithmDetails)
         #Save Trained models
        saveModel(trainedModel,clf_Name)
    #   Evaluate model using voting classifier     
    ensembleModel(algorithmDetails,x_Train,x_Test,y_Train,y_Test)       
    #Plot and save accuracy plot and accuracy score
    plotAndSaveAccuracyScore(algorithmDetails)
    #Plot confusion matrix
    plotConfusionMatrix(algorithmDetails)
    #Save classification Report
    save_classification_report(algorithmDetails)
###########################################################################################
#   Function        :   ensembleModel
#   Input Params    :   algorithmDetails(Dictionary holding all details),x_Train,x_Test,y_Train,y_Test
#   Output Params   :   None
#   Description     :   Voting classifier for both models
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
##########################################################################################
def ensembleModel(algorithmDetails,x_Train,x_Test,y_Train,y_Test): 
    clf_Name= "Voting Classifier"
    
    print("Training ensemble model(Voting Classifier)")
    algorithmDetails["Algorithm Name"].append(clf_Name)
    #MODEL_NAMES = list(CLF_MODELS.items())
    ensemblemodel=VotingClassifier(
        estimators=[("RF",algorithmDetails["TrainedModel"][0]),
                    ("GB",algorithmDetails["TrainedModel"][1])],
        voting="soft",
        weights=[1,1])
    #   Build Pipeline
    #pipelineModel=build_Pipeline(ensembleModel)
    #   Train model
    trainedModel=trainModel(ensemblemodel,x_Train,y_Train)
    algorithmDetails["TrainedModel"].append(trainedModel) 
    # Model accuracy calculations
    testModelAndAccuracyCalculation(trainedModel,x_Test,y_Test,algorithmDetails)
    #   Save Trained models
    saveModel(trainedModel,"Voting Classifier")    
###########################################################################################
#   Function        :   save_classification_report
#   Input Params    :   algorithmDetails(Dictionary holding all details)
#   Output Params   :   Saves classification report
#   Description     :   prints and saves classification report with precision,recall,F1 score
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
##########################################################################################
def save_classification_report(algorithmDetails):
    out_dir=ARTIFACT_DIR
    ensure_dir(out_dir)
   
    reportData=""
    for cnt in range(len(algorithmDetails["Classification Report"])):
        reportData=f"{reportData}\t{algorithmDetails["Algorithm Name"][cnt]}\n{BORDER}"
        clsReport=algorithmDetails["Classification Report"][cnt]
        reportData=f"{reportData}\n{clsReport}{BORDER}\n\n"
        reportData=f"{reportData}\nAccuracy Score : {algorithmDetails['Accuracy Score'][cnt]}\n{BORDER}\n\n"

    with open (os.path.join(out_dir,"classification_report.txt"),"w") as f:
        f.write(reportData)         
###########################################################################################
#   Function        :   plotConfusionMatrix
#   Input Params    :   Dictionary holding all details
#   Output Params   :   -
#   Description     :   Saves confusion matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
############################################################################################    
def plotConfusionMatrix(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    """Create the figure and subplots
    As we used 2 algorithms use Subplots to plot 1 confusion matrix"""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 8)) 
    for cnt in range(len(algorithmDetails["Confusion Matrix"])):
        
        confuMatrix = ConfusionMatrixDisplay(confusion_matrix=algorithmDetails["Confusion Matrix"][cnt])
        confuMatrix.plot(ax=axes[cnt])
        axes[cnt].set_title(algorithmDetails["Algorithm Name"][cnt])
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"ConfusionMatrix.png"))
    plt.close()               
###########################################################################################
#   Function        :   plotAndSaveAccuracyScore
#   Input Params    :   Dictionary holding all details
#   Output Params   :   -
#   Description     :   Plot and saves accuracy score
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
############################################################################################    
def plotAndSaveAccuracyScore(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    x=algorithmDetails['Algorithm Name']
    y=algorithmDetails['Accuracy Score']
    plt.figure(figsize=(8,7))
    plt.bar(x,y,width=0.2)
    plt.title('Accuracy of algorithms')
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.yticks(y)
    plt.grid(True)
    #plt.show()
    outDir=os.path.join(ARTIFACT_DIR,"AccuracyPlot.png")
    plt.savefig(outDir)
    plt.close()
    print(f"Saved Accuracy Plot :{outDir}")             
#####################################################################################################
#   Function name    :  testModelAndAccuracyCalculation
#   Input Params     :  model,xTrain,yTrain data set,algorithmDetails(Dictionary)
#   Output           :  None
#   Description      :  Prediction and Accuracy calculations
#   Author           :  Vaishali M Jorwekar
#   Date             :  19 Dec 2025
#####################################################################################################
def testModelAndAccuracyCalculation(model,x_Test,y_Test,algorithmDetails):
    """Predict the test results"""
    model_Predicted=model.predict(x_Test)
    """Accuracy calculations"""
    model_Accuracy=accuracy_score(y_Test,model_Predicted)
    """Calculate confusion Matrix """
    model_ConfusionMatrix=confusion_matrix(y_Test,model_Predicted)
    """Model classification report"""
    model_Classification_Report=classification_report(y_Test,model_Predicted)

    algorithmDetails["Accuracy Score"].append(model_Accuracy*100)
    algorithmDetails["Confusion Matrix"].append(model_ConfusionMatrix)
    algorithmDetails["Classification Report"].append(model_Classification_Report)       
###########################################################################################
#   Function        :   train_and_evaluate
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Training of model
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
############################################################################################
def train_and_evaluate():
    #   Pre-Process Data
    dFrame=preProcessData() 
     #   Apply TF-IDF vectorizer
    tfv=TfidfVectorizer(stop_words="english",max_features=5000)
    tfv.fit(dFrame["cleanedText"])
    tfidf_matrix=tfv.transform(dFrame["cleanedText"])
     #   Save Matrices
    saveModel(tfv,"tfidf_matrix")
    #   Train Model
    trainAndFindAccuracy(dFrame,tfidf_matrix)
    
    
#########################################################################################################
#   Function Name    :  sentimentAnalysis
#   Description      :  main function sentimentAnalysis
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  19 Dec 2025
#########################################################################################################
def sentimentAnalysis():
     ensure_dir(ARTIFACT_DIR)
     train_and_evaluate()
     
#########################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  19 Dec 2025
#########################################################################################################
def main():
    sentimentAnalysis()

#########################################################################################################
#   Starter
#########################################################################################################
if __name__=="__main__":
    main()