"""-----------------------------------------------------------------------------------------------------
                        Breast Cancer Prediction
                    (Student name - Vaishali Jorwekar)
                    Python by Marvellous Infosystems
--------------------------------------------------------------------------------------------------------
Problem statement: Based on given information find whether given tumor is 
                   malignant or benign
--------------------------------------------------------------------------------------------------------"""
#####################################################################################################
# Required Python Packages
#####################################################################################################
import pandas as pd
import os,numpy as np
import argparse
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,confusion_matrix,classification_report,
    ConfusionMatrixDisplay,roc_curve,auc
)
import joblib
#####################################################################################################
#   Constants
#####################################################################################################

ARTIFACT_DIR="artifact_BC"
BORDER="-"*65
TEST_SIZE=0.2
RANDOM_STATE=42
TARGET_COLNAME="target"
#Classification models
CLF_MODELS={"Logistic Regression":LogisticRegression(),
            "SVM":SVC(probability=True,kernel='linear')}

###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training,testing
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
############################################################################################
def parse_args(): 
    p=argparse.ArgumentParser(description="Breast cancer prediction case study")
    p.add_argument("--train",action="store_true",
                   help="Training using 'Logistic Regression' and 'Support Vector Machine(SVM)'")
    p.add_argument("--test",action="store_true",
                   help="Test the models using saved and pretrained models")
    p.add_argument("--samples",type=int,default=10,help="Number of samples for testing")
    return p.parse_args()
#####################################################################################################
#   Function Name   :   LoadAndExploreDataset()
#   Description     :   Load and explore data set
#                       Load data using pandas
#                       Handle missing or unknown values
#                       Display basic statistics and visualise class distibution
#   Input Params    :   -
#   Output          :   Data Frame
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
####################################################################################################
def LoadAndExploreDataset():
    df=datasets.load_breast_cancer(as_frame=True)
    data=df.data
    target=df.target
    print("data:",data)

    print("target:",target)
    """Dataset statistics"""
    df=displayDatasetStatistics(df)
    return df

####################################################################################################
#   Function Name   :   displayDatasetStatistics
#   Description     :   This method loads the data set information
#   Input params    :   Data frame
#   Output          :   Data statistics
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
#####################################################################################################
def displayDatasetStatistics(df):
    """Show column info and check for null values
    ------------------------------------------------""" 
    data=df.data
    target=df.target
    df=pd.concat([data,target],axis=1)
    df.dropna(inplace=True)

    print(BORDER)
    """Display basic statistics using .describe()
    ------------------------------------------------"""
    print("Data set stastistics")
    print(BORDER)
    print(df.describe())
    print(BORDER)

    print("Data set columns details...")
    print(BORDER)
    print(df.columns)
    return df
###########################################################################################
#   Function        :   preProcessData
#   Input Params    :   None
#   Output Params   :   df(Data Frame)
#   Description     :   Load CSV data and pre-process data
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
############################################################################################
def preProcessData():
    #Load data from datasets
    df=LoadAndExploreDataset()
    """Encode Data set"""
    EncodeDataSet(df)
    print(df.head())
    """load co-relation matrix"""
    displayCorrelationMatrix(df)
    """Drop invalid records"""
    df.dropna(inplace=True)
    return df
#####################################################################################################
#   Function Name   :   displayCorrelationMatrix
#   Description     :   Plot the Co-relation matrix
#   Input Params    :   Data frame   
#   Output          :   Co-relation Matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
#####################################################################################################
def displayCorrelationMatrix(df):
    ensure_dir(ARTIFACT_DIR)
    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
    plt.title("Feature co-relation heatmap")
    plt.savefig(os.path.join(ARTIFACT_DIR,"Co-Relation.png"))
#####################################################################################################
#   Function Name   :   EncodeDataSet
#   Description     :   Prepare data by applying label encoding
#   Input params    :   data frame
#   Output          :   Creating encoded csv file for refernce with name 'DataEncoded.csv' 
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
#####################################################################################################
def EncodeDataSet(df):
    #Label enconding for all data set coulmns
    labelEncoder = LabelEncoder()
    for colName in df.select_dtypes(include=['object']).columns: 
        df[colName]=labelEncoder.fit_transform(df[colName])   
        df[colName].unique()
    print(BORDER)
    print("Creating encoded csv file for refernce with name 'DataEncoded.csv'")
    df.to_csv(os.path.join(ARTIFACT_DIR,"DataEncoded.csv"))
    print(BORDER)
    print("Encoded Data frame")
    print(BORDER)
    print(df.head())
    print(BORDER)
    return df
#####################################################################################################
#   Function name    :  trainPipeline
#   Description      :  Train a pipeline
#   Input Params     :  pipeline,xTrain,yTrain data set
#   Output           :  Return pipeline object
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
#####################################################################################################
def trainPipeline(pipeline,xTrain,yTrain):
    pipeline.fit(xTrain,yTrain)
    return pipeline 
#####################################################################################################
#   Function name    :  testModelAndAccuracyCalculation
#   Input Params     :  pipeline,xTrain,yTrain data set,algorithmDetails(Dictionary)
#   Output           :  -
#   Description      :  Accuracy Calculations
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
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
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training,testing
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
############################################################################################
def train_and_evaluate():
     #Load CSV and pre-process data
    df=preProcessData()
    #Split data set into training and testing set
    xTrain,xTest,yTrain,yTest=split_DataSet(df)
    algorithmDetails={"Algorithm Name":[],"Training Accuracy":[],
                      "Accuracy Score":[],
                     "Confusion Matrix":[],"Classification Report":[]}    
    rocAucCurveDetails={}
    
    rocAucCurveDetails["Actual"]=yTest
    featureImportanceList=[]
    for clfName,clfModel in CLF_MODELS.items():
        #Build pipeline
        pipeline=build_Pipeline(clfModel)
        #Train pipeline
        trainedModel=trainPipeline(pipeline,xTrain,yTrain)
        print(f"Trained pipeline for {clfName}: {trainedModel}")
        algorithmDetails["Algorithm Name"].append(clfName)
        # result prediction and accuracy calculations        
        testModelAndAccuracyCalculation(trainedModel,xTest,yTest,algorithmDetails)
        #Calculate ROC Details
        calculateROC(trainedModel,xTest,yTest,clfName,rocAucCurveDetails)
        # Save the trained model
        saveTrainedModel(trainedModel,clfName)
        
        #Plot feature Importance
        featureName=df.drop(columns=TARGET_COLNAME).columns
        findModelFeatureImportance(trainedModel,featureName,clfName,featureImportanceList)
    #Plot and save accuracy plot and accuracy score
    plotAndSaveAccuracyScore(algorithmDetails)
    #Plot confusion matrix
    plotConfusionMatrix(algorithmDetails)
    #Save classification Report
    save_classification_report(algorithmDetails)
    #Plot feature importance model wise
    featureImportnancesModelWise(featureImportanceList)
    #Plot ROC-AUC Curve
    plotROCAUCCurve(rocAucCurveDetails,algorithmDetails)
#####################################################################################################
#   Function name    :  calculateROC
#   Input Params     :  pipeline,xTrain,yTrain data set,algorithmDetails(Dictionary)
#   Output           :  -
#   Description      :  Calculate ROC-AUC Details
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
#####################################################################################################
def calculateROC(model,xTest,yTest,clfName,rocAucCurveDetails):
    """Positive Predictions"""
    y_predPositiveClass = model.predict_proba(xTest)[:, 1]
    """ROC-AUC Score"""
    rocAucCurveDetails[clfName]=y_predPositiveClass
#####################################################################################################
#   Function Name   :   plotROCAUCCurve
#   Input Params    :   rocAucCurveDetails(Dictionary)
#   Description     :   Plot ROC-AUC Curve model wise
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
# #####################################################################################################
def plotROCAUCCurve(rocAucCurveDetails,algorithmDetails): 
    """ROC Curve using sub plots"""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,8)) 

    for algoCnt in range(len(algorithmDetails['Algorithm Name'])):
        algoName=algorithmDetails['Algorithm Name'][algoCnt]
       
        
        falsePositiveRate, truePositiveRate, threshold = \
        roc_curve(rocAucCurveDetails['Actual'], rocAucCurveDetails[algoName])
        
        rocAuc = auc(falsePositiveRate, truePositiveRate)
        axes[algoCnt].plot(falsePositiveRate,truePositiveRate,
                           label=f"{algoName} (ROC-AUC Score {rocAuc})")
        axes[algoCnt].plot([0, 1], [0, 1], color="Red", label='Base Line',linestyle='--')

        axes[algoCnt].set_xlabel('False Positive Rate')
        axes[algoCnt].set_ylabel('True Positive Rate')
        axes[algoCnt].set_title(f'ROC Curves : {algorithmDetails["Algorithm Name"]}')
        #confuMatrix.plot(ax=axes[algoName])
        axes[algoCnt].set_title(algorithmDetails["Algorithm Name"][algoCnt])
        axes[algoCnt].legend()

    plt.tight_layout()
    #plt.show()  
    ensure_dir(ARTIFACT_DIR)
    outDir=os.path.join(ARTIFACT_DIR,"ROC-AUC Curve.png") 
    plt.savefig(outDir)
#####################################################################################################
#   Function Name   :   featureImportnancesModelWise
#   Input Params    :   featureImportanceList(Model wise)
#   Description     :   Plotting feature Importnace model wise for comparision

#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
# #####################################################################################################
def featureImportnancesModelWise(featureImportanceList):   
    ensure_dir(ARTIFACT_DIR)
    """Create the figure and subplots
    As we used 2 algorithms use Subplots to plot 2 feature importance matrix"""
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
#   Date            :   31 oct 2025
##########################################################################################
def save_classification_report(algorithmDetails):
    out_dir=ARTIFACT_DIR
    ensure_dir(out_dir)
    reportData=""
    for cnt in range(len(algorithmDetails["Classification Report"])):
        reportData=f"{reportData}\t{algorithmDetails["Algorithm Name"][cnt]}\n{BORDER}"
        clsReport=algorithmDetails["Classification Report"][cnt]
        reportData=f"{reportData}\n{clsReport} \n \
            Training Accuracy:{algorithmDetails["Accuracy Score"][cnt]}\n{BORDER}\n\n"
    with open (os.path.join(out_dir,"classification_report.txt"),"w") as f:
        f.write(reportData)      
###########################################################################################
#   Function        :   plotConfusionMatrix
#   Input Params    :   Dictionary holding all details
#   Output Params   :   -
#   Description     :   Saves confusion matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
############################################################################################    
def plotConfusionMatrix(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    """Create the figure and subplots
    For 2 algorithms use Subplots to plot 1 confusion matrix"""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5)) 
    for cnt in range(len(algorithmDetails["Confusion Matrix"])):
            confuMatrix = ConfusionMatrixDisplay(
                confusion_matrix=algorithmDetails["Confusion Matrix"][cnt])
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
#   Date            :   31 Oct 2025
############################################################################################    
def plotAndSaveAccuracyScore(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    x=algorithmDetails["Algorithm Name"]
    y=algorithmDetails["Accuracy Score"]  

    plt.figure(figsize=(10,8))
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
#   Function name    :  saveTrainedModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
#####################################################################################################
def saveTrainedModel(model,modelName):
    ensure_dir(ARTIFACT_DIR)
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)
#####################################################################################################
#   Function name    :  findModelFeatureImportance
#   Description      :  finds feature importance
#   Input Params     :  model,feature names,model name,featureImportanceList
#   Output           :  Plotted output
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
#####################################################################################################
def findModelFeatureImportance(model,feature_names,modelName,featureImportanceList):  
    importances=[]
    featureImportance={"Feature Algorithm Name":"",
                       "Feature Names":[],
                       "Importances":[]}

    if hasattr(model,"named_steps") and "clf" in model.named_steps:
        clf=model.named_steps["clf"]
        if hasattr(clf,"feature_importances_"):
            importances=(clf.feature_importances_)
        elif hasattr(clf,"coef_"):
            importances=clf.coef_[0] 
            importances=np.abs((importances))
    else:
        print("Feature importance is not available for this model")
        return
    featureImportance["Feature Algorithm Name"]=modelName
    featureImportance["Feature Names"]=feature_names
    featureImportance["Importances"]=importances
    featureImportanceList.append(featureImportance)
#####################################################################################################
#   Function name    :  build_Pipeline
#   Description      :  Build a pipeline
#   Input Params     :  colTransformer,clf_Name
#   Output           :  Return pipeline object
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
#####################################################################################################
def build_Pipeline(clf_Name):
    pipe=Pipeline(steps=[
        ("scalar",StandardScaler()),
        ("clf",clf_Name)
    ])
    return pipe
#####################################################################################################
#   Function Name   :   split_DataSet
#   Description     :   This method splits the data set into training and testing sets
#   Input           :   data frame
#   Output          :   Testing and Training features and Target    
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025   
#####################################################################################################
def split_DataSet(df):
    """Divide data set into features and target"""
    features,target=findFeaturesAndTarget(df)
    xTrain,xTest,yTrain,yTest=train_test_split(features,
                                               target,
                                               test_size=TEST_SIZE,
                                               random_state=RANDOM_STATE)
    return xTrain,xTest,yTrain,yTest

#####################################################################################################
#   Function Name   :   findFeaturesAndTarget
#   Description     :   Finds Target and Features
#   Input Params    :   Data Frame
#   Output          :   Features and Targets  
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025   
#####################################################################################################
def findFeaturesAndTarget(df):
    features=df.drop(columns=[TARGET_COLNAME])
    target=df[TARGET_COLNAME]
    return features,target    
#####################################################################################################
#   Function Name    :  BCPrediction 
#   Description      :  Manages calls to training and testing functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
#####################################################################################################
def BCPrediction():
    ensure_dir(ARTIFACT_DIR)
    args=parse_args()
    did_anything=False
    if args.train:
        train_and_evaluate()
        did_anything=True
    if args.test:
        test_models(n_samples=args.samples)
        did_anything=True    
    if not did_anything:
        print(
            "Nothing to do .Try one of the :\n"
            "python3 BC_Prediction.py --train\n"
            "python3 BC_Prediction.py --test\n"
        )    
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
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
#   Date            :   31 Oct 2025
#####################################################################################################
def testTrainedModel(model,sampleTestData): 
     predictedResult=model.predict(sampleTestData.drop(columns=[TARGET_COLNAME]))
     predictionAccuracy=accuracy_score(sampleTestData[TARGET_COLNAME],predictedResult)
     return predictionAccuracy,predictedResult          
###########################################################################################
#   Function        :   plotConfusionMatrix
#   Input Params    :   Number of samples
#   Output Params   :   -
#   Description     :   Test Model using test data
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
############################################################################################    
def test_models(n_samples):   
    df=preProcessData()
    sampleTestData =df.sample(n=n_samples)
    resultMapping = {0: "Yes(Cancer)", 1: "No (Non Cancerous)"}
    #result=['Yes(Cancerous)','No (Non-Cancerous)']
    #Copy test data
    copyTestData=sampleTestData.copy()
    print(BORDER)
    print(f"Selected sample data for testing are \n: {sampleTestData}")
    print(BORDER)
    """Loading saved model and sample tetsing"""
    testResult=f"Test Data :\n{BORDER}\n{copyTestData}\n{BORDER}\n"
    predictedResultsDF=pd.DataFrame()
    predictedResultsDF[TARGET_COLNAME]=copyTestData[TARGET_COLNAME]
    for clf_Name,clfModel in CLF_MODELS.items():  
        print(f"Sample testing Using : {clf_Name}")  
        # Load the model and test samples
        trainedModel=loadTrainedModel(clf_Name)
        # Test Model on a random sample from data set
        predictionAccuracy,predictedResult=testTrainedModel(trainedModel,sampleTestData)
       
        predictedResultsDF[clf_Name]=predictedResult
        testResult=f"\n{testResult}{clf_Name}\nTesting Accuracy:{predictionAccuracy*100}\n{BORDER}\n"
    pd.set_option('future.no_silent_downcasting', True)
    predictedResultsDF=predictedResultsDF.replace(resultMapping)  
    testResult=f"{testResult}{predictedResultsDF}"    
    outDir=os.path.join(ARTIFACT_DIR,"PredictedResult.txt")
    with open(outDir,"w") as f:
         f.write(str(testResult))    
#####################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Oct 2025
#####################################################################################################
def main():
    BCPrediction()
#---------------------------------------------------------------------------------------------------------
# Main entry point of the application
#---------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    main()
