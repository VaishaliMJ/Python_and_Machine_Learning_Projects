"""-----------------------------------------------------------------------------------------------------
                    Head Brain co-relation case study
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement: Built Decision Tree Classifier model to analyse the correlation 
                   between Gender and brain size,brain weight
--------------------------------------------------------------------------------------------------------"""
#####################################################################################################
# Required Python Packages
#####################################################################################################
import os,argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
import joblib
#####################################################################################################
# Constants and file names
#####################################################################################################
BORDER="-"*65
DATASET_FILENAME="MarvellousHeadBrain.csv"
ARTIFACT_DIR="artifact_HeadBrainClassification"
HEAD_SIZE="Head Size(cm^3)"
BRAIN_WEIGHT="Brain Weight(grams)"
TARGET_COLUMN="Gender"
TEST_SIZE=0.2
RANDOM_STATE=42
MODEL_NAME="Decision Tree Classifier"
###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,testing
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
############################################################################################
def parse_args():
    p=argparse.ArgumentParser(description="Gender,Head Brain Co-Relation Case Study")
    p.add_argument("--train",action="store_true",
                   help="Train Using 'Decision Tree Classifier' and Save Artifacts")
    p.add_argument("--test",action="store_true",help="Test the model using saved model")
    p.add_argument("--samples",type=int,default=10,help="Number of samples for testing ")
    return p.parse_args() 
###########################################################################################
#   Function        :   preProcessData
#   Input Params    :   None
#   Output Params   :   df(Data Frame)
#   Description     :   Load CSV data and pre-process data
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
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
#   Function        :   displayCoRelationMatrix
#   Input Params    :   dFrame(data frame)
#   Output Params   :   -
#   Description     :   Displays co-relation matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
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
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Oct 2025
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
#   Date            :   10 Oct 2025
############################################################################################
def cleanDataSet(dFrame)->pd.DataFrame:
    print(BORDER)
    print("Cleaning Data set....")
    print(BORDER)
    print("Dimensions of data set before cleaning is :",dFrame.shape)
    #Remove unwanted data
    dFrame.dropna(inplace=True)
    print(BORDER)
    print("Data set after cleaning")
    print(dFrame.head())
    return dFrame
#####################################################################################################
#   Function name    :  build_Pipeline
#   Input Params     :  clf_Name: model name
#   Output           :  Return pipeline object
#   Description      :  Build a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
#####################################################################################################
def build_Pipeline(clf_Name):
    pipe=Pipeline(steps=[
         ("scalar",StandardScaler()),
         ("clf",clf_Name),
    ])
    return pipe 
#####################################################################################################
#   Function name    :  trainPipeline
#   Input Params     :  pipeline,xTrain,yTrain data set
#   Output           :  Return pipeline object
#   Description      :  Train a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
#####################################################################################################
def trainPipeline(pipeline,xTrain,yTrain):
    pipeline.fit(xTrain,yTrain)
    return pipeline     

###########################################################################################
#   Function        :   train_and_evaluate
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training,testing
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
############################################################################################
def train_and_evaluate():
    #Load CSV and pre-process data
    df=preProcessData()
    #Split data set  
    x_Train,x_Test,y_Train,y_Test=spliDataSet(df)
     #Decision Tree Classifier
    dTree=build_Pipeline(DecisionTreeClassifier(max_depth=20))
     #Train Model
    dTree=trainPipeline(dTree,x_Train,y_Train)
    # Test model
    dTreeAccuracy,dTreeConfusionMatrix,dTreeClassificationReport\
                            =testModelAndAccuracyCalculation(dTree,x_Test,y_Test)
    
    #Plot confusion matrix
    plotConfusionMatrix(dTreeConfusionMatrix)
    #Save classification Report
    save_classification_report(dTreeClassificationReport,dTreeAccuracy)
    #Save Trained models
    saveTrainedModel(dTree,MODEL_NAME)
###########################################################################################
#   Function        :   save_classification_report
#   Input Params    :   dTreeClassificationReport,dTreeAccuracy
#   Output Params   :   Saves classification report
#   Description     :   prints and saves classification report with precision,recall,F1 score
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
##########################################################################################
def save_classification_report(dTreeClassificationReport,dTreeAccuracy):
    out_dir=ARTIFACT_DIR
    ensure_dir(out_dir)
    reportData=""
    reportData=f"{reportData}\tAlgorithm Name:Decision Tree Classifier\n{BORDER}"
    reportData=f"{reportData}\n{dTreeClassificationReport}{BORDER}\n\n"

    reportData=f"{reportData}\nAccuracy Of Model: {dTreeAccuracy}\n{BORDER}\n\n"
    with open (os.path.join(out_dir,"classification_report.txt"),"w") as f:
        f.write(reportData)      
###########################################################################################
#   Function        :   plotConfusionMatrix
#   Input Params    :   dTreeConfusionMatrix
#   Output Params   :   -
#   Description     :   Saves confusion matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   1 Oct 2025
############################################################################################    
def plotConfusionMatrix(dTreeConfusionMatrix):
    ensure_dir(ARTIFACT_DIR)
        
    confuMatrix = ConfusionMatrixDisplay(confusion_matrix=dTreeConfusionMatrix)
    confuMatrix.plot()

    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"ConfusionMatrix.png"))
    plt.close()    
#####################################################################################################
#   Function name    :  testModelAndAccuracyCalculation
#   Input Params     :  pipeline,xTrain,yTrain data set
#   Output           :  Return pipeline object
#   Description      :  Train a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
#####################################################################################################
def testModelAndAccuracyCalculation(model,x_Test,y_Test):
    """Predict the test results"""
    model_Predicted=model.predict(x_Test)
    """Accuracy calculations"""
    model_Accuracy=accuracy_score(y_Test,model_Predicted)
    """Calculate confusion Matrix """
    model_ConfusionMatrix=confusion_matrix(y_Test,model_Predicted)
    """Model classification report"""
    model_Classification_Report=classification_report(y_Test,model_Predicted)
    return model_Accuracy,model_ConfusionMatrix,model_Classification_Report    
###########################################################################################
#   Function        :   plotHeadSizeVsBrainWt
#   Input Params    :   x_Test(Head Size),y_Test(Brain Weight),y_Predicted(Predicted Brain Weight)
#   Output Params   :   -
#   Description     :   Plots the Head Size vs Brain Weigth Plot
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
############################################################################################ 
def plotHeadSizeVsBrainWt(x_Test,y_Test,predictedBrainWt):
    plt.figure(figsize=(10,8))
    plt.scatter(x_Test,y_Test,color="blue",label="Actual")
    plt.plot(x_Test.values.flatten(),predictedBrainWt,color="red",linewidth=2,label="Regression Line")
    plt.xlabel("Head Size(cm^3)")
    plt.ylabel("Brain Weight(grams)")
    plt.title("Head Size Vs Head Brain Regression Plot")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"HeadSizeVsBrainWtRegressionPlot.png"))

###########################################################################################
#   Function        :   spliDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
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
#   Description     :   This method spilts data set into features and target
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
############################################################################################
def findFeaturesAndTarget(dFrame):
    xFeatures=dFrame.drop(columns=[TARGET_COLUMN])
    yTarget=dFrame[TARGET_COLUMN]
    print(BORDER)    
    return xFeatures,yTarget
#####################################################################################################
#   Function name    :  saveTrainedModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
#####################################################################################################
def saveTrainedModel(model,modelName):
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)
    print(f"Model saved to path :{path}")        
###########################################################################################
#   Function        :   test_Model
#   Input Params    :   n_samples(Number of samples)
#   Output Params   :   -
#   Description     :   Test the model
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
############################################################################################    
def test_Model(n_samples):
    df=preProcessData()
    sampleTestData =df.sample(n=n_samples)
    #Copy test data
    copyTestData=sampleTestData.copy()
    print(BORDER)
    print(f"Selected sample data for testing are \n: {sampleTestData}")
    print(BORDER)
    print(f"Sample testing Using :'{MODEL_NAME}'")  
    # Load the model and test samples
    trainedModel=loadTrainedModel(MODEL_NAME)
    predicteResult=testTrainedModel(trainedModel,sampleTestData)
    # Save testing results
    save_Test_Results(copyTestData,predicteResult)
#####################################################################################################
#   Function name    :  save_Test_Results
#   Input Params     :  dataFrame,PredictedResults
#   Output           :  model
#   Description      :  Test the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   10 Oct 2025
#####################################################################################################
def save_Test_Results(dataFrame,PredictedResults):
    dataFrame["Predicted Result"]=PredictedResults
    outDir=os.path.join(ARTIFACT_DIR,"PredictedResult.txt")
    with open(outDir,"w") as fErr:
         fErr.write(str(dataFrame))
    
#####################################################################################################
#   Function name    :  testTrainedModel
#   Input Params     :  path = MODEL_PATH,n_samples
#   Output           :  model
#   Description      :  Test the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Oct 2025
#####################################################################################################
def testTrainedModel(model,sampleTestData): 
     predictedResult=model.predict(sampleTestData.drop(columns=[TARGET_COLUMN]))
     return predictedResult        
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Oct 2025
#####################################################################################################
def loadTrainedModel(modelName):  
    path=modelName+".joblib"
    path=os.path.join(ARTIFACT_DIR,path)
    model = joblib.load(path)
    print(f"Model loaded from the path :{path}")
    return model  
#####################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   8 Oct 2025
#####################################################################################################
def main():
    ensure_dir(ARTIFACT_DIR)
    did_anything=False
    args=parse_args()
    #Train the model
    if args.train:
        train_and_evaluate()
        did_anything=True
    if args.test:
        test_Model(n_samples=args.samples)
        did_anything=True   
    if not did_anything:
        print(
            "Nothing to do .Try one of the :\n"
            "python3 HeadBrainClassification.py --train\n"
            "python3 HeadBrainClassification.py--test\n"
        )           
#####################################################################################################
#     Starter
#####################################################################################################
if __name__=="__main__":
    main()