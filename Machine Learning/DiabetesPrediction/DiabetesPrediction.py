"""-----------------------------------------------------------------------------------------------------
                          Diabetes Prediction
                        Vaishali Jorwekar
--------------------------------------------------------------------------------------------------------
Problem statement:Dataset contains some medical parameters based on that predict 
                 for diabetes positive or negative with classification algorithms
                 Decision Tree and Logistic Regression
--------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports 
#####################################################################################################
import pandas as pd
import os,argparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
#####################################################################################################
# Constants and file names
#####################################################################################################
BORDER="-"*65
DATASET_FILENAME="diabetes.csv"
ARTIFACT_DIR="artifact_Diabetes"
"""Classification model list"""
CLF_MODELS=  {"Logistic Regression":LogisticRegression(),
              "KNN": KNeighborsClassifier(),
             "Decision Tree Classifier":DecisionTreeClassifier()}
TARGET_COLUMN="Outcome"
TEST_SIZE=0.2
RANDOM_STATE=42

###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)

###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,interference baselines
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
############################################################################################
def parse_args():
    p=argparse.ArgumentParser(description="Diabetes Prediction Case Study")
    p.add_argument("--train",action="store_true",
                   help="Traing using Logistic Regression,KNN and Decision Tree")

    p.add_argument("--test",action="store_true",help="Test the models using saved models")
    p.add_argument("--samples",type=int,default=10,help="Number of samples for testing ")
    return p.parse_args()
###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
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
#   Function        :   cleanDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   Updated data frame 
#   Description     :   Cleans data set by removing unwanted columns
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
############################################################################################
def cleanDataSet(dFrame)->pd.DataFrame:
    print(BORDER)
    print("Cleaning Data set....")
    print(BORDER)
    print("Dimensions of data set before cleaning is :",dFrame.shape)
    dFrame=missingValProcessing(dFrame)
    return dFrame
#####################################################################################################
#  Function Name : missingValProcessing
#  Description   : Check and handle missing or zero values in columns
#  Input params  : dfDiabetes(Data frame set)
#  Output        : Replace '0' values with column mean
#####################################################################################################
def missingValProcessing(dfDiabetes):
    ensure_dir(ARTIFACT_DIR)
    fileData=""
    fileData=f"{fileData}\t\tZero values in column report"
    fileData=f"{fileData}\n{BORDER}"
    replaceZeroValCols=[]
    missingValuesDF={"ColName":[],"Missing Value Count":[]}
    for colName in dfDiabetes.drop(columns=TARGET_COLUMN).columns:
        missingValCount=(dfDiabetes[colName]==0).sum()
        missingValuesDF['ColName'].append(colName)
        missingValuesDF["Missing Value Count"].append(missingValCount)
        if(missingValCount>0):
            replaceZeroValCols.append(colName)
    """Displaying statistics of missing values count vs Column name"""
    missingValDataFrame=pd.DataFrame(missingValuesDF)
    fileData=f"{fileData}\n{missingValDataFrame}"

    fileData=f"{fileData}\nColumn List with zero values"
    fileData=f"{fileData}\n{replaceZeroValCols}"
    fileData=f"{fileData}\nReplacing each 0 column values with column mean"
    for colName in replaceZeroValCols:
        """Replace '0' with column mean"""
        dfDiabetes[colName]=dfDiabetes[colName].replace(0,dfDiabetes[colName].mean())
    #fileData=f"{fileData}\n{dfDiabetes}"
    #fileData=f"{fileData}\n{dfDiabetes.shape}"

    with open(os.path.join(ARTIFACT_DIR,"missing_Col_Values_Report.txt"),"w") as f:
        f.write(fileData)
    return dfDiabetes
###########################################################################################
#   Function        :   displayCoRelationMatrix
#   Input Params    :   dFrame(data frame)
#   Output Params   :   -
#   Description     :   Displays co-relation matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Sep 2025
############################################################################################
def displayCoRelationMatrix(dFrame):
    print(BORDER)
    ensure_dir(ARTIFACT_DIR)
    plt.figure(figsize=(10,6))
    sns.heatmap(dFrame.corr(),annot=True,cmap="coolwarm")
    plt.title("Feature co-relation heatmap")
    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"CoRelationMatrix.png"))
###########################################################################################
#   Function        :   spliDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
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
#   Date            :   12 Oct 2025
############################################################################################
def findFeaturesAndTarget(dFrame):
    xFeatures=dFrame.drop(columns=[TARGET_COLUMN])
    yTarget=dFrame[TARGET_COLUMN]
    print(BORDER)    
    return xFeatures,yTarget
###########################################################################################
#   Function        :   preProcessData
#   Input Params    :   None
#   Output Params   :   df(Data Frame)
#   Description     :   Load CSV data and pre-process data
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
############################################################################################
def preProcessData():
    #Load data from CSV file
    df=readCSVFile(csvFileName=DATASET_FILENAME)
    #Clean data set
    df=cleanDataSet(df)
    #Print co-relation matrix
    displayCoRelationMatrix(df)
    return df
###########################################################################################
#   Function        :   train_and_evaluate
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training,testing
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
############################################################################################
def train_and_evaluate():
    #Load CSV and pre-process data
    df=preProcessData()
    #Split data set
    x_Train,x_Test,y_Train,y_Test=spliDataSet(df)
    featureImportanceList=[]
    algorithmDetails={"Algorithm Name":[],"TrainedModel":[],"Accuracy Score":[],
                      "Confusion Matrix":[],"Classification Report":[]}
    """Creating pipelines for 3 models"""    
    for clf_Name,clfModel in CLF_MODELS.items():
        #3. Build and Train pipeline
         pipeline=build_Pipeline(clfModel) 
         trainedModel=trainPipeline(pipeline,x_Train,y_Train)
         print(f"Trained pipeline for {clf_Name}")
         algorithmDetails["Algorithm Name"].append(clf_Name)
         algorithmDetails["TrainedModel"].append(trainedModel)
         # Model accuracy calculations
         testModelAndAccuracyCalculation(trainedModel,x_Test,y_Test,algorithmDetails)
         
         #Save Trained models
         saveTrainedModel(trainedModel,clf_Name)
         if(clf_Name!="KNN"):   
             featureName=df.drop(columns=TARGET_COLUMN).columns
             #print(featureName)
             plotFeatureImportance(trainedModel,featureName,clf_Name,featureImportanceList)
    #Plot and save accuracy plot and accuracy score
    plotAndSaveAccuracyScore(algorithmDetails)
    #Plot confusion matrix
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
#   Date            :   12 Oct 2025
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
#####################################################################################################
#   Function name    :  saveTrainedModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
#####################################################################################################
def saveTrainedModel(model,modelName):
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)
    print(f"Model saved to path :{path}")    
###########################################################################################
#   Function        :   save_classification_report
#   Input Params    :   algorithmDetails(Dictionary holding all details)
#   Output Params   :   Saves classification report
#   Description     :   prints and saves classification report with precision,recall,F1 score
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Sep 2025
##########################################################################################
def save_classification_report(algorithmDetails):
    out_dir=ARTIFACT_DIR
    ensure_dir(out_dir)
    reportData=""
    for cnt in range(len(algorithmDetails["Classification Report"])):
        reportData=f"{reportData}\t{algorithmDetails["Algorithm Name"][cnt]}\n{BORDER}"
        clsReport=algorithmDetails["Classification Report"][cnt]
        reportData=f"{reportData}\n{clsReport}{BORDER}\n\n"
    with open (os.path.join(out_dir,"classification_report.txt"),"w") as f:
        f.write(reportData)  
###########################################################################################
#   Function        :   plotConfusionMatrix
#   Input Params    :   Dictionary holding all details
#   Output Params   :   -
#   Description     :   Saves confusion matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Sep 2025
############################################################################################    
def plotConfusionMatrix(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    """Create the figure and subplots
    As we used 3 algorithms use Subplots to plot 1 confusion matrix"""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5)) 
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
#   Date            :   12 Oct 2025
############################################################################################    
def plotAndSaveAccuracyScore(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    x=algorithmDetails['Algorithm Name']
    y=algorithmDetails['Accuracy Score']
    plt.figure(figsize=(8,7))
    plt.bar(x,y,width=0.4)
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
#   Input Params     :  pipeline,xTrain,yTrain data set,algorithmDetails(Dictionary)
#   Output           :  Return pipeline object
#   Description      :  Train a pipeline
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
    #return model_Accuracy,model_ConfusionMatrix,model_Classification_Report
#####################################################################################################
#   Function name    :  trainPipeline
#   Input Params     :  pipeline,xTrain,yTrain data set
#   Output           :  Return pipeline object
#   Description      :  Train a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Sep 2025
#####################################################################################################
def trainPipeline(pipeline,xTrain,yTrain):
    pipeline.fit(xTrain,yTrain)
    return pipeline  
#####################################################################################################
#   Function name    :  build_Pipeline
#   Input Params     :  clf_Name: model name
#   Output           :  Return pipeline object
#   Description      :  Build a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Sep 2025
#####################################################################################################
def build_Pipeline(clf_Name):
    pipe=Pipeline(steps=[
        ("scalar",StandardScaler()),
        ("clf",clf_Name)
    ])  
    return pipe 
#####################################################################################################
#   Function name    :  plotFeatureImportance
#   Description      :  Plots feature importance
#   Input Params     :  model
#   Output           :  Plotted output
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
#####################################################################################################
def plotFeatureImportance(model,feature_names,modelName,featureImportanceList):
    importances=[]
    featureImportance={"Feature Algorithm Name":"",
                       "Feature Names":[],
                       "Importances":[]}
    
    if hasattr(model,"named_steps") and "clf" in model.named_steps:
        clf=model.named_steps['clf']
        if hasattr(clf,"feature_importances_"):
            importances=(clf.feature_importances_)
        elif hasattr(clf,"coef_"):
             importances=clf.coef_[0] 
             importances=np.abs((importances))
    else:
        print("Feature importance is not available for this model")
        return
    """elif(hasattr(model,"feature_importances_")):
        importances=model.feature.importances_
    elif(hasattr(model,"coef_")):
         importances=model.coef_[0] 
         importances=np.abs((importances))
         #print(f"Importances :{importances}")
     """
    featureImportance["Feature Algorithm Name"]=modelName
    featureImportance["Feature Names"]=feature_names
    featureImportance["Importances"]=importances
    featureImportanceList.append(featureImportance)
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Sep 2025
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
#   Date            :   12 Oct 2025
#####################################################################################################
def testTrainedModel(model,sampleTestData): 
     predictedResult=model.predict(sampleTestData.drop(columns=[TARGET_COLUMN]))
     predictionAccuracy=accuracy_score(sampleTestData[TARGET_COLUMN],predictedResult)
     return predictionAccuracy,predictedResult
###########################################################################################
#   Function        :   plotConfusionMatrix
#   Input Params    :   Dictionary holding all details
#   Output Params   :   -
#   Description     :   Saves confusion matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
############################################################################################    
def test_models(n_samples):
    df=preProcessData()
    sampleTestData =df.sample(n=n_samples)
    resultMapping = {0: "Non-diabetes", 1: "Diabetes"}
    #Copy test data
    copyTestData=sampleTestData.copy()
    print(BORDER)
    print(f"Selected sample data for testing are \n: {sampleTestData}")
    print(BORDER)
    """Loading saved model and sample tetsing"""
    testResult=f"Test Data :\n{BORDER}\n{copyTestData}\n{BORDER}\n"
    predictedResultsDF=pd.DataFrame()
    predictedResultsDF[TARGET_COLUMN]=copyTestData[TARGET_COLUMN]
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
#   Function Name    :  DiabetesPrediction 
#   Description      :  Manages calls to training and testing functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
#####################################################################################################
def DiabetesPrediction():
    ensure_dir(ARTIFACT_DIR)
    args=parse_args()
    did_anything=False
    #Model training
    if args.train:
        train_and_evaluate()
        did_anything=True
    #Model testing    
    if args.test:
        test_models(n_samples=args.samples)
        did_anything=True    
    if not did_anything:
        print(
            "Nothing to do .Try one of the :\n"
            "python3 DiabetesPrediction.py --train\n"
            "python3 DiabetesPrediction.py --test\n"
        )    

#####################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Oct 2025
#####################################################################################################
def main():
    DiabetesPrediction()
#---------------------------------------------------------------------------------------------------------
# Main entry point of the application
#---------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    main()