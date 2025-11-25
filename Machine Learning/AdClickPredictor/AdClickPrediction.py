"""-----------------------------------------------------------------------------------------------------
                         Ad click predictor
                    (Student name - Vaishali Jorwekar)
                    Python : Marvellous Infosystems
--------------------------------------------------------------------------------------------------------
Problem statement: Classification model predicting probability of user clicks on advertisement
--------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Required Python Packages
#####################################################################################################
import os,argparse,numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
#####################################################################################################
#   Constants
#####################################################################################################
ARTIFACT_DIR="artifact_AdClickPredictor"
DATASET_FILENAME="Ad Click Data.csv"
BORDER="-"*65
TEST_SIZE=0.2
RANDOM_STATE=42
#Model name
CLF_MODELS={"Logistic Regression": LogisticRegression(),
            "Decision Tree Classifier":DecisionTreeClassifier(),
            "KNN":KNeighborsClassifier()
            }

TARGET_COLNAME='Clicked on Ad'
FEATURES=['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage']

#####################################################################################################
#   Function Name   :   ensure_dir
#   Input Params    :   path of directory
#   Output Params   :   None
#   Description     :   Check and create ARTIFACT_DIR if does not exists
#   Author          :   Vaishali M. Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def ensure_dir(path:str):
    os.makedirs(ARTIFACT_DIR,exist_ok=True)
#####################################################################################################
#   Function Name   :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,testing baselines
#   Author          :   Vaishali M. Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def parse_args():
    p=argparse.ArgumentParser(description="Ad Click Prediction Case Study")
    p.add_argument("--train",action="store_true",
                   help="Training using regression models")
    p.add_argument("--test",action="store_true",
                   help="Testing of pre-trained models")
    p.add_argument("--samples",type=int,default=5,help="Number of samples for testing ")
    return p.parse_args()  
###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
############################################################################################
def readCSVFile(csvFileName=DATASET_FILENAME)->pd.DataFrame:
    dFrame=pd.read_csv(csvFileName)
    print(BORDER)
    print(f"Data loaded successfully from file '{csvFileName}'")
    print(BORDER)
    print(f"File Data:\n{BORDER}\n{dFrame.describe()}")
    print(BORDER)
    return dFrame
###########################################################################################
#   Function        :   analyseDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   Updated data frame 
#   Description     :   Analyse data set for Duplicate and 'na' values
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
############################################################################################
def analyseDataSet(dFrame)->pd.DataFrame:
    print(dFrame.info())
    print("\n",dFrame.describe)
    print(BORDER)
    print("Cleaning Data set....")
    print(BORDER)
    print("Dimensions of data set before cleaning is :",dFrame.shape)
    print(BORDER)
    print("Checking for 'na' values:")
    print(BORDER)
    print(dFrame.isna().sum())
    print(BORDER)
    print("Checking for 'Duplicate' values:")
    print(dFrame.duplicated().sum())
    return dFrame
#####################################################################################################
#   Function Name   :   preProcessData
#   Input Params    :   Directory Name
#   Output Params   :   None
#   Description     :   Train and evaluate the models
#   Author          :   Vaishali M. Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def preProcessData():
    df=pd.DataFrame()
    try:
        #   Read CSV file
        df=readCSVFile(csvFileName=DATASET_FILENAME)
    except FileNotFoundError:
        print(f"File : {DATASET_FILENAME} not found !!!")   
        exit()    
    #   Clean dataset
    df=analyseDataSet(df)
    #   Co-Relation matrix 
    displayCoRelationMatrix(df)
    #   Clean Data set
    df_train=cleanDataSet(df)
    return df,df_train
###########################################################################################
#   Function        :   cleanDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   Updated data frame 
#   Description     :   Clean Data set using Imputer
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
############################################################################################
def cleanDataSet(dFrame)->pd.DataFrame:
    # Convert Timestamp column into day,month and year column
    dFrame['Timestamp']=pd.to_datetime(dFrame['Timestamp'])
    dFrame['Month']=dFrame['Timestamp'].dt.month
    dFrame['Day']=dFrame['Timestamp'].dt.day

    dFrame['Weekday']=dFrame['Timestamp'].dt.dayofweek

    dFrame['Hour']=dFrame['Timestamp'].dt.hour

    dFrame=dFrame.drop(['Timestamp'],axis=1)

    print(dFrame.head())
    #   Filling null values with Meadian values
    imputer=SimpleImputer()
    dFrame_numeric=dFrame.select_dtypes(exclude='object')
    imputedDF=imputer.fit_transform(dFrame_numeric)

    #convert back into pandas data frame
    df_train=pd.DataFrame(imputedDF)
    df_train.columns=dFrame.select_dtypes(exclude='object').columns

    
    #   Scale features
    scalar=StandardScaler()
    df_train[FEATURES]=scalar.fit_transform(df_train[FEATURES])

    return df_train

###########################################################################################
#   Function        :   displayCoRelationMatrix
#   Input Params    :   dFrame(data frame)
#   Output Params   :   -
#   Description     :   Displays co-relation matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
############################################################################################
def displayCoRelationMatrix(dFrame):
    print(BORDER)
    ensure_dir(ARTIFACT_DIR)
    
    plt.figure(figsize=(18,10))
    #   Selected features
    selectedCols=['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male','Clicked on Ad']
    sns.heatmap(dFrame[selectedCols].corr(),annot=True,cmap="coolwarm")
    plt.title("Feature co-relation heatmap")
    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"CoRelationMatrix.png"))  
#####################################################################################################
#   Function name    :  trainModel
#   Input Params     :  model,xTrain,yTrain data set
#   Output           :  Return model object
#   Description      :  Train a model
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def trainModel(model,xTrain,yTrain):
    model.fit(xTrain,yTrain)
    return model       
#####################################################################################################
#   Function Name   :   trainAndEvaluate
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Train and evaluate the models
#   Author          :   Vaishali M. Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def trainAndEvaluate():  
     #   Pre-Process data
    df,df_train=preProcessData()
    #   Split data set
    x_Train,x_Test,y_Train,y_Test=spliDataSet(df_train)
    print(f"x_Test:{x_Test}")
    print(f"y_Test:{y_Test}")
    algorithmDetails={"Algorithm Name":[],"TrainedModel":[],"Accuracy Score":[],
                      "Confusion Matrix":[],"Classification Report":[]}
    featureImportanceList=[]
    for clf_Name,clf_model in CLF_MODELS.items():
        trainedModel=trainModel(clf_model,x_Train,y_Train)

        algorithmDetails["Algorithm Name"].append(clf_Name)
        algorithmDetails["TrainedModel"].append(trainedModel)

        #Model accuracy calculations
        testModelAndAccuracyCalculation(trainedModel,x_Test,y_Test,algorithmDetails)

        #Save Trained models
        saveTrainedModel(trainedModel,clf_Name)
        
        if(clf_Name!="KNN"):   
            featureName=df.drop(columns=TARGET_COLNAME).columns

            plotFeatureImportance(trainedModel,featureName,clf_Name,featureImportanceList)

    #Plot and save accuracy plot and accuracy score
    plotAndSaveAccuracyScore(algorithmDetails)

    #Plot confusion Matrix
    plotConfusionMatrix(algorithmDetails)

    #Save classification Report
    save_classification_report(algorithmDetails)

    #Plot feature importance model wise
    featureImportnancesModelWise(featureImportanceList)
#####################################################################################################
#   Function Name   :   featureImportnancesModelWise
#   Input Params    :   featureImportanceList(Model wise)
#   Description     :   Plotting feature Importnace model wise for comparision

#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
# #####################################################################################################
def featureImportnancesModelWise(featureImportanceList):   
    ensure_dir(ARTIFACT_DIR)
    """Create the figure and subplots
    As we used 3 algorithms use Subplots to plot 3 confusion matrix"""
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,8)) 

    for cnt in range(len(featureImportanceList)):
        modelName=featureImportanceList[cnt]["Feature Algorithm Name"]
        feature_names=featureImportanceList[cnt]["Feature Names"]
        importances=featureImportanceList[cnt]["Importances"]
        idx=np.argsort((importances))[::-1]
    
        
        axes[cnt].bar(range(len(importances)),importances[idx])
        axes[cnt].set_xticks(range(len(importances)),[feature_names[i] for i in idx],rotation =50,ha='right')
        axes[cnt].set_ylabel("Importance")
        axes[cnt].set_title(f"Feature Importance Plot : {modelName}")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR,"FeatureImportance.png"))
    plt.close()    
###########################################################################################
#   Function        :   save_classification_report
#   Input Params    :   algorithmDetails(Dictionary holding all details)
#   Output Params   :   Saves classification report
#   Description     :   prints and saves classification report with precision,recall,F1 score
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
##########################################################################################
def save_classification_report(algorithmDetails):
    out_dir=ARTIFACT_DIR 
    ensure_dir(out_dir)
    reportData=""
    for cnt in range(len(algorithmDetails["Classification Report"])):
        reportData=f"{reportData}\n{algorithmDetails["Algorithm Name"][cnt]}\n{BORDER}"
        clsReport=algorithmDetails["Classification Report"][cnt]
        reportData=f"{reportData}\n{clsReport}{BORDER}\n\n"
    with open (os.path.join(ARTIFACT_DIR,"ClassificationReport.txt"),"w") as f:
        f.write(reportData)

###########################################################################################
#   Function        :   plotConfusionMatrix
#   Input Params    :   Dictionary holding all details
#   Output Params   :   -
#   Description     :   Plots and Saves confusion matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
############################################################################################    
def plotConfusionMatrix(algorithmDetails):  
    ensure_dir(ARTIFACT_DIR)
    """Create the figure and subplots
    As we used 3 algorithms use Subplots to plot 1 confusion matrix"""
    fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
    for cnt in range(len(algorithmDetails["Confusion Matrix"])):
        confuMatrix=ConfusionMatrixDisplay(confusion_matrix=algorithmDetails["Confusion Matrix"][cnt])
        confuMatrix.plot(ax=axes[cnt],)
        axes[cnt].set_title(algorithmDetails["Algorithm Name"][cnt])
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR,"ConfusionMatrix.png"))
    plt.close()        

###########################################################################################
#   Function        :   plotAndSaveAccuracyScore
#   Input Params    :   Dictionary holding all details
#   Output Params   :   -
#   Description     :   Plot and saves accuracy score
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
############################################################################################    
def plotAndSaveAccuracyScore(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    x=algorithmDetails["Algorithm Name"] 
    y=algorithmDetails["Accuracy Score"]
    plt.figure(figsize=(10,8))  
    plt.bar(x,y,width=0.4)
    plt.title("Accuracy Plot of Algorithms")
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")
    plt.yticks(y)
    plt.grid(True)
    outDir=os.path.join(ARTIFACT_DIR,"AccuracyPlot.png")
    plt.savefig(outDir) 
    plt.close()
    print(f"Saved Accuracy Plot :{outDir}")
    
#####################################################################################################
#   Function name    :  plotFeatureImportance
#   Description      :  Plots feature importance
#   Input Params     :  model
#   Output           :  Plotted output
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def plotFeatureImportance(model,feature_names,modelName,featureImportanceList):
    importances=[]
    featureImportance={"Feature Algorithm Name":"",
                       "Feature Names":[],
                       "Importances":[]}
    
    if hasattr(model,"feature_importances_"):
            importances=(model.feature_importances_)
    elif hasattr(model,"coef_"):
            importances=model.coef_[0] 
            importances=np.abs((importances))
    else:
        print("Feature importance is not available for this model")
        return
    
    featureImportance["Feature Algorithm Name"]=modelName
    featureImportance["Feature Names"]=feature_names
    featureImportance["Importances"]=importances
    featureImportanceList.append(featureImportance)        
#####################################################################################################
#   Function name    :  saveTrainedModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def saveTrainedModel(model,modelName):
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)
    print(f"Model saved to path :{path}")             
#####################################################################################################
#   Function Name   :   testModelAndAccuracyCalculation
#   Input Params    :   model,x_Test,y_Test,algorithmDetails
#   Output Params   :   None
#   Description     :   Train and evaluate the models
#   Author          :   Vaishali M. Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def testModelAndAccuracyCalculation(model,x_Test,y_Test,algorithmDetails):  
    """Predict the test results"""
    model_Predicted=model.predict(x_Test)

    
    #   Accuracy calculations
    model_Accuracy=accuracy_score(y_Test,model_Predicted)
    algorithmDetails["Accuracy Score"].append(model_Accuracy*100)

    #   Confusion Matrix
    model_CM=confusion_matrix(y_Test,model_Predicted)
    algorithmDetails["Confusion Matrix"].append(model_CM)

    #   Classification Report
    model_CR=classification_report(y_Test,model_Predicted)
    algorithmDetails["Classification Report"].append(model_CR)


###########################################################################################
#   Function        :   spliDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
############################################################################################
def spliDataSet(dFrame):
    xFeatures,yTarget=findFeaturesAndTarget(dFrame)
    x_Train,x_Test,y_Train,y_Test=train_test_split(xFeatures,
                                                   yTarget,
                                                   test_size=TEST_SIZE,
                                                   random_state=RANDOM_STATE)
    
    return x_Train,x_Test,y_Train,y_Test   
###########################################################################################
#   Function        :   findFeaturesAndTarget
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
############################################################################################
def findFeaturesAndTarget(dFrame):
    
    xFeatures=findFeatures(dFrame)
    #xFeatures=dFrame.drop(columns=[TARGET_COLNAME])
    
    yTarget=dFrame[TARGET_COLNAME]
    print(BORDER)    
    return xFeatures,yTarget 
###########################################################################################
#   Function        :   findFeatures
#   Input Params    :   dFrame(data frame)
#   Output Params   :   Feature Extraction
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
############################################################################################
def findFeatures(dFrame):
    #   Selected features
    train_Features=['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male',
          'Month', 'Day', 'Weekday', 'Hour']
    xFeatures=dFrame[train_Features]
    return xFeatures
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author           :  Vaishali M Jorwekar
#   Date             :  18 Nov 2025
#####################################################################################################
def loadTrainedModel(modelName):  
    path=modelName+".joblib"
    path=os.path.join(ARTIFACT_DIR,path)
    model = joblib.load(path)
    print(f"Model loaded from the path :{path}")
    return model 
#####################################################################################################
#   Function name    :  testTrainedModel
#   Input Params     :  path = MODEL_PATH,n_samples
#   Output           :  model
#   Description      :  Test the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def testTrainedModel(model,sampleTestData): 
     predictedResult=model.predict(sampleTestData.drop(columns=[TARGET_COLNAME]))
     predictionAccuracy=accuracy_score(sampleTestData[TARGET_COLNAME],predictedResult)
     return predictionAccuracy,predictedResult  
###########################################################################################
#   Function        :   test_Models
#   Input Params    :   Number of samples
#   Output Params   :   -
#   Description     :   Test Model using test data
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2025
############################################################################################    
def test_Models(n_samples):
    df,dFrame=preProcessData()
    sampleTestData=dFrame.sample(n=n_samples)
    resultMapping = {0 :"user didn't click", 1:"user clicked"}
    #Copy test data
    copyTestData=sampleTestData.copy()
    print(BORDER)
    print(f"Selected sample data for testing are \n: {sampleTestData}")
    print(BORDER)
    """Loading saved model and sample tetsing"""
    testResult=f"Test Data :\n{BORDER}\n{copyTestData}\n{BORDER}\n"
    predictedResults_DF=pd.DataFrame()
    predictedResults_DF[TARGET_COLNAME]=copyTestData[TARGET_COLNAME]
    for clf_Name,clfModel in CLF_MODELS.items(): 
        print(f"Sample testing Using : {clf_Name}")  
        # Load the model and test samples
        trainedModel=loadTrainedModel(clf_Name) 

         # Test Model on a random sample from data set
        predictionAccuracy,predictedResult=testTrainedModel(trainedModel,sampleTestData)
        predictedResults_DF[clf_Name]=predictedResult
        testResult=f"\n{testResult}{clf_Name}\nTesting Accuracy:{predictionAccuracy*100}\n{BORDER}\n"
    pd.set_option('future.no_silent_downcasting', True)
    predictedResults_DF=predictedResults_DF.replace(resultMapping)  
    testResult=f"{testResult}{predictedResults_DF}"    
    outDir=os.path.join(ARTIFACT_DIR,"PredictedResult.txt")
    with open(outDir,"w") as f:
         f.write(str(testResult))       
#####################################################################################################
#   Function Name   :   adClickPredictor
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def adClickPredictor():
    ensure_dir(ARTIFACT_DIR)
    did_anything=False
    args=parse_args()
    if args.train:
        trainAndEvaluate()
        did_anything=True
    if args.test:
        test_Models(n_samples=args.samples)   
        did_anything=True 
    if not did_anything:
        print(
            "Nothing to do .Try one of the :\n"
            "python3 AdClickPrediction.py --train\n"
            "python3 AdClickPrediction.py --test\n"
        )    
#####################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar
#   Date            :   18 Nov 2025
#####################################################################################################
def main():
    adClickPredictor()

#####################################################################################################
#   Starter
#####################################################################################################
if __name__=="__main__":
    main()