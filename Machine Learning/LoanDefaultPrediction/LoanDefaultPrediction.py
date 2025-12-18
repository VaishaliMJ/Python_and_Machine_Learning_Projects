"""---------------------------------------------------------------------------------------------------------------
                Loan Default Prediction (Ensemble-Random Forest & Gradient Boosting)
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Boosted model accuracy by combining multiple classifiers to predict loan repayment defaults
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import os,argparse,joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
#####################################################################################################
# Constants and file names
#####################################################################################################
BORDER="-"*65
DATASET_FILENAME="Loan_default.csv"
ARTIFACT_DIR="ARTIFACT_LoanDefault"
TARGET_COLUMN="Default"
TEST_SIZE=0.2
RANDOM_STATE=42
NUM_ESTIMATORS=100
LEARNING_RATE=0.1

"""Classification model list"""
CLF_MODELS={"Random Forest Classifier":RandomForestClassifier(n_estimators=NUM_ESTIMATORS),
            "Gradient Boosting Classifier":GradientBoostingClassifier(n_estimators=NUM_ESTIMATORS,
                                                                      learning_rate=LEARNING_RATE)}
##########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Dec 2025
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,testing
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Dec 2025
############################################################################################
def parse_args():   
    p=argparse.ArgumentParser(description="Loan Default Prediction Case Study")
    p.add_argument("--train",action="store_true",help="Traing using Random Forest & Gradient Boosting")
    p.add_argument("--test",action="store_true",help="Test the models using saved models")
    p.add_argument("--samples",type=int,default=10,help="Number of samples for testing ")
    return p.parse_args()
###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Dec 2025
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
#   Date            :   8 Dec 2025
###########################################################################################
def cleanDataSet(dFrame):    
    print(BORDER)
    print("Cleaning Data set....")
    print(BORDER)
    print(f"Null value check:\n{dFrame.isna().sum()}")
    print("Dimensions of data set before cleaning is :",dFrame.shape)
    print(f"'na' value count\n:{BORDER}\n{dFrame.isna().sum()}")
    print(f"Dropping 'na' value records:")
    dFrame.dropna(inplace=True)
    print("Dimensions of data set after removing 'na' values:",dFrame.shape)
    print(f"Data Set Info:\n{dFrame.info}")
    dFrame.drop(["LoanID"],axis=1,inplace=True)
    return dFrame
#####################################################################################################
#   Function Name   :   EncodeDataSet
#   Description     :   Prepare data by applying label encoding
#   Input params    :   data frame
#   Output          :   Creating encoded csv file for refernce with name 'DataEncoded.csv' 
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Dec 2025
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
#   Function Name   :   displayCorrelationMatrix
#   Description     :   Plot the Co-relation matrix
#   Input Params    :   Data frame   
#   Output          :   Co-relation Matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Dec 2025
#####################################################################################################
def displayCorrelationMatrix(dFrame):
    ensure_dir(ARTIFACT_DIR)
    #   Find Object columns and apply LabelEncoding
    df=EncodeDataSet(dFrame)
    #   Co-Relation matrix
    plt.figure(figsize=(15,10))
    sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
    plt.title("Feature co-relation heatmap")
    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"CoRelationMatrix.png"))
    plt.close()
    return df
###########################################################################################
#   Function        :   preProcessData
#   Input Params    :   None
#   Output Params   :   df(Data Frame)
#   Description     :   Load CSV data and pre-process data
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Dec 2025
############################################################################################
def preProcessData():
    #Load data from CSV file
    df=readCSVFile(csvFileName=DATASET_FILENAME)
     #   Clean data set
    df=cleanDataSet(df)
    #   EDA
    plotFeaturesDistribution(df)
     #   Display co-realation matrix
    displayCorrelationMatrix(df)
    
    return df
###########################################################################################
#   Function        :   plotFeaturesDistribution
#   Input Params    :   dFrame(data frame)
#   Output Params   :   None
#   Description     :   This method Plots different features
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
############################################################################################
def plotFeaturesDistribution(dFrame):
    
   
    sns.countplot(x="Default",data=dFrame)
    plt.title("Default distribution")
    plt.xlabel("Default Status")
    plt.ylabel("Count")
    plt.savefig(os.path.join(ARTIFACT_DIR,"DefaulterAnalysis.png"))
    plt.close()
    
     
    sns.countplot(x="Age",data=dFrame,hue='Default')
    plt.title("Age wise Defaulter")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR,"AgeWiseAnalysis.png"))
    plt.close()
    
    
    fig,axes=plt.subplots(ncols=2,nrows=2,figsize=(20,18)) 
    
    sns.countplot(x="LoanPurpose",hue='Default',data=dFrame,ax=axes[0][0])
    axes[0][0].set_title("Loan Purpose wise Defaulter")
    axes[0][0].set_xlabel("Loan Purpose")
    axes[0][0].set_ylabel("Frequency")
    #axes[0][0].legend()
    
    sns.countplot(x="EmploymentType",hue='Default',data=dFrame,ax=axes[0][1],width=0.4)
    axes[0][1].set_title("Employment Type Defaulter")
    axes[0][1].set_xlabel("Default Status")
    axes[0][1].set_ylabel("Count")
    #axes[0][1].legend()
    
    sns.countplot(x="Education",hue='Default',data=dFrame,ax=axes[1][0],width=0.4)
    axes[1][0].set_title("Education Wise Defaulter")
    axes[1][0].set_xlabel("Education")
    axes[1][0].set_ylabel("Count")
    #axes[1][0].legend(loc="best")
    
    sns.countplot(data=dFrame, x='MaritalStatus', hue='Default',ax=axes[1][1])

    axes[1][1].set_title("Marital Status Wise Defaulter")
    axes[1][1].set_xlabel("Marital Status")
    axes[1][1].set_ylabel("Count")
    #axes[1][1].legend(loc="best")
    
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR,"DataAnalysis.png"))
    plt.close()
    
    
###########################################################################################
#   Function        :   findFeaturesAndTarget
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
############################################################################################
def findFeaturesAndTarget(dFrame):
    xFeatures=dFrame.drop(columns=[TARGET_COLUMN])
    yTarget=dFrame[TARGET_COLUMN]
    print(BORDER)    
    return xFeatures,yTarget
###########################################################################################
#   Function        :   spliDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
############################################################################################
def spliDataSet(dFrame):
    xFeatures,yTarget=findFeaturesAndTarget(dFrame)
   
    x_Train,x_Test,y_Train,y_Test=train_test_split(xFeatures,
                                                   yTarget,
                                                   test_size=TEST_SIZE,
                                                   random_state=RANDOM_STATE)
    
    return x_Train,x_Test,y_Train,y_Test
#####################################################################################################
#   Function name    :  build_Pipeline
#   Input Params     :  clf_Name: model name
#   Output           :  Return pipeline object
#   Description      :  Build a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
#####################################################################################################
def build_Pipeline(clf_Name):
    pipe=Pipeline(steps=[
        ("scalar",StandardScaler()),
        ("clf",clf_Name)
    ])  
    return pipe 
#####################################################################################################
#   Function name    :  trainPipeline
#   Input Params     :  pipeline,xTrain,yTrain data set
#   Output           :  Return pipeline object
#   Description      :  Train a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
#####################################################################################################
def trainPipeline(pipeline,xTrain,yTrain):
    pipeline.fit(xTrain,yTrain)
    return pipeline  
###########################################################################################
#   Function        :   train_and_evaluate
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Training of model
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Dec 2025
############################################################################################
def train_and_evaluate():
    #Load CSV and pre-process data
    df=preProcessData()
    
    algorithmDetails={"Algorithm Name":[],"TrainedModel":[],"Accuracy Score":[],
                      "Confusion Matrix":[],"Classification Report":[]}
    
    #Split data set
    featureImportanceList=[]
    x_Train,x_Test,y_Train,y_Test=spliDataSet(df)
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
         featureName=df.drop(columns=TARGET_COLUMN).columns

         plotFeatureImportance(trainedModel,featureName,clf_Name,featureImportanceList)

    #   Evaluate model using voting classifier     
    ensembleModel(algorithmDetails,x_Train,x_Test,y_Train,y_Test)     
    #Plot and save accuracy plot and accuracy score
    plotAndSaveAccuracyScore(algorithmDetails)
    #Plot confusion matrix
    plotConfusionMatrix(algorithmDetails)
    #Save classification Report
    save_classification_report(algorithmDetails)
    #Plot feature importance model wise
    featureImportnancesModelWise(featureImportanceList)

#####################################################################################################
#   Function name    :  plotFeatureImportance
#   Description      :  Plots feature importance
#   Input Params     :  model
#   Output           :  Plotted output
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
#####################################################################################################
def plotFeatureImportance(model,feature_names,modelName,featureImportanceList):
    importances=[]
    
    featureImportance={"Feature Algorithm Name":"",
                       "Feature Names":[],
                       "Importances":[]}
    importances=[]
    if hasattr(model,"named_steps") and "clf" in model.named_steps:
        clf=model.named_steps["clf"]
        importances=clf.feature_importances_   
    elif(hasattr(model,"feature.importances_")):
            importances=model.feature.importances_
    else:
        print("Feature importance is not available for this model")
        return
    
    featureImportance["Feature Algorithm Name"]=modelName
    featureImportance["Feature Names"]=feature_names
    featureImportance["Importances"]=importances
    featureImportanceList.append(featureImportance) 
#####################################################################################################
#   Function Name   :   featureImportnancesModelWise
#   Input Params    :   featureImportanceList(Model wise)
#   Description     :   Plotting feature Importnace model wise for comparision

#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
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
#   Function        :   ensembleModel
#   Input Params    :   algorithmDetails(Dictionary holding all details),x_Train,x_Test,y_Train,y_Test
#   Output Params   :   None
#   Description     :   Voting classifier for both models
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
##########################################################################################
def ensembleModel(algorithmDetails,x_Train,x_Test,y_Train,y_Test): 
    clf_Name= "Voting Classifier"
    
    print("Training ensemble model(Voting Classifier)")
    algorithmDetails["Algorithm Name"].append(clf_Name)
    #MODEL_NAMES = list(CLF_MODELS.items())
    ensemblemodel=VotingClassifier(
        estimators=[("RF",algorithmDetails["TrainedModel"][0]),("GB",algorithmDetails["TrainedModel"][1])],
        voting="soft",
        weights=[1,1])
    #   Build Pipeline
    #pipelineModel=build_Pipeline(ensembleModel)
    #   Train model
    trainedModel=trainPipeline(ensemblemodel,x_Train,y_Train)
    
    
    algorithmDetails["TrainedModel"].append(trainedModel) 
    # Model accuracy calculations
    testModelAndAccuracyCalculation(trainedModel,x_Test,y_Test,algorithmDetails)
          #Save Trained models
    saveTrainedModel(trainedModel,"Voting Classifier")

###########################################################################################
#   Function        :   save_classification_report
#   Input Params    :   algorithmDetails(Dictionary holding all details)
#   Output Params   :   Saves classification report
#   Description     :   prints and saves classification report with precision,recall,F1 score
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
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
#   Date            :   12 Dec 2025
############################################################################################    
def plotConfusionMatrix(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    """Create the figure and subplots
    As we used 3 algorithms use Subplots to plot 1 confusion matrix"""
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
#   Date            :   12 Dec 2025
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
#   Function name    :  saveTrainedModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
#####################################################################################################
def saveTrainedModel(model,modelName):
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)
    print(f"Model saved to path :{path}")             
#####################################################################################################
#   Function name    :  testModelAndAccuracyCalculation
#   Input Params     :  pipeline,xTrain,yTrain data set,algorithmDetails(Dictionary)
#   Output           :  Return pipeline object
#   Description      :  Train a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
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
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
#####################################################################################################
def loadTrainedModel(modelName):  
    path=modelName+".joblib"
    path=os.path.join(ARTIFACT_DIR,path)
    model = joblib.load(path)
    print(f"Model loaded from the path :{path}")
    return model          
###########################################################################################
#   Function        :   test_Models
#   Input Params    :   Number of samples
#   Output Params   :   -
#   Description     :   Test Model using test data
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
############################################################################################    
def test_models(n_samples):
    df=preProcessData()
    sampleTestData =df.sample(n=n_samples)
    resultMapping = {0: "Non-Deafulter", 1: "Defaulter"}
    #Copy test data
    copyTestData=sampleTestData.copy()
    print(BORDER)
    print(f"Selected sample data for testing are \n: {sampleTestData}")
    print(BORDER)
    """Loading saved model and sample tetsing"""
    testResult=f"Test Data :\n{BORDER}\n{copyTestData}\n{BORDER}\n"
    predictedResultsDF=pd.DataFrame()
    predictedResultsDF[TARGET_COLUMN]=copyTestData[TARGET_COLUMN]
    MODEL_NAMES=["Random Forest Classifier","Gradient Boosting Classifier","Voting Classifier"]
                                             
    for cnt in range(len(MODEL_NAMES)):  
        clf_Name=MODEL_NAMES[cnt]
        print(f"Sample testing Using : {clf_Name}")  
        # Load the model and test samples
        trainedModel=loadTrainedModel(clf_Name)
        # Test Model on a random sample from data set
        predictionAccuracy,predictedResult=testTrainedModel(trainedModel,sampleTestData)
       
        predictedResultsDF[clf_Name]=predictedResult
        testResult=f"\n{testResult}{clf_Name}\nTesting Accuracy:{predictionAccuracy*100}\n{BORDER}\n"
    pd.set_option('future.no_silent_downcasting', True)
    pd.set_option('display.width', 9999)
    pd.set_option('display.max_columns', None)
    predictedResultsDF=predictedResultsDF.replace(resultMapping)  
    testResult=f"{testResult}{predictedResultsDF}"    
    outDir=os.path.join(ARTIFACT_DIR,"PredictedResult.txt")
    
    with open(outDir,"w") as f:
         f.write(str(testResult))   
#####################################################################################################
#   Function name    :  testTrainedModel
#   Input Params     :  path = MODEL_PATH,n_samples
#   Output           :  model
#   Description      :  Test the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Dec 2025
#####################################################################################################
def testTrainedModel(model,sampleTestData): 
     predictedResult=model.predict(sampleTestData.drop(columns=[TARGET_COLUMN]))
     predictionAccuracy=accuracy_score(sampleTestData[TARGET_COLUMN],predictedResult)
     return predictionAccuracy,predictedResult                     
#########################################################################################################
#   Function Name    :  loanDefaultPrediction  
#   Description      :  Traing and testing function call
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   8 Dec 2025
#########################################################################################################
def loanDefaultPrediction():
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
            "python3 LoanDefaultPrediction.py --train\n"
            "python3 LoanDefaultPrediction.py --test\n"
        )    
#########################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Dec 2025
#########################################################################################################
def main():
    loanDefaultPrediction()

#########################################################################################################
#   Starter
#########################################################################################################
if __name__=="__main__":
    main()

