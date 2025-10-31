"""-----------------------------------------------------------------------------------------------------
                       Iris Classification
                        Vaishali Jorwekar
--------------------------------------------------------------------------------------------------------
Problem statement:Implement classic ML classification on the Iris dataset using 
                  Decision Tree and KNN
--------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports 
#####################################################################################################
import pandas as pd,os
from sklearn.datasets import load_iris
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ( accuracy_score,confusion_matrix,
                             classification_report,ConfusionMatrixDisplay)
import joblib

#####################################################################################################
#   Constants
#####################################################################################################
ARTIFACT_DIR="artifact_Iris"
DATASET_FILENAME="iris.csv"
BORDER="-"*65
TARGET_COLUMN="variety"
TEST_SIZE=0.2
RANDOM_STATE=42
"""Classification models"""
CLF_MODELS={"Decision Tree Classifier":DecisionTreeClassifier(max_depth=10),
            "KNN":KNeighborsClassifier(n_neighbors=3)}

###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,testing
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
############################################################################################
def parse_args():
    p=argparse.ArgumentParser(description="Iris Classification Case Study")
    p.add_argument("--train",action="store_true",
                   help="Traing using KNN and Decision Tree")
    p.add_argument("--test",action="store_true",help="Test the models using saved models")
    p.add_argument("--samples",type=int,default=10,help="Number of samples for testing ")
    return p.parse_args()
###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
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
#   Date            :   30 Oct 2025
############################################################################################
def cleanDataSet(dFrame)->pd.DataFrame:
    print(BORDER)
    print("Cleaning Data set....")
    print(BORDER)
    print("Dimensions of data set before cleaning is :",dFrame.shape)
    dFrame.fillna(0, inplace=True)
   
    return dFrame

######################################################################################################
#   Function        :   EncodeDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   Updated data frame 
#   Description     :   Prepare data by applying label encoding
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025 
######################################################################################################
def EncodeDataSet(dFrame):
    ensure_dir(ARTIFACT_DIR)
    #Label enconding for all data set coulmns
    for colName in dFrame.select_dtypes(include=['object']).columns:
        #print(colName)
        labelEncoder = LabelEncoder()
        dFrame[colName]=labelEncoder.fit_transform(dFrame[colName])
        dFrame[colName].unique()
    print(BORDER)
    print("Creating encoded csv file for refernce with name 'IrisDataEncoded.csv'")
    dFrame.to_csv(os.path.join(ARTIFACT_DIR,"IrisDataEncoded.csv"))
    print(BORDER)
    print("Encoded Data frame")
    print(BORDER)
    print(dFrame.head())
    print(BORDER)
    return dFrame
###########################################################################################
#   Function        :   preProcessData
#   Input Params    :   None
#   Output Params   :   df(Data Frame)
#   Description     :   Load data set and pre-process data
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
############################################################################################
def preProcessData():
    #Load data from CSV file
    df=readCSVFile(csvFileName=DATASET_FILENAME)
    #Missing value report
    findMissingValues(df)

    #Clean data set
    df=cleanDataSet(df)
    # Scatter Plot of classification
    plotIrisVariety(df)
    #Encode Data set
    df=EncodeDataSet(df)
    #Print co-relation matrix
    displayCoRelationMatrix(df)
    return df
###########################################################################################
#   Function        :   findMissingValues
#   Input Params    :   df(Data Frame)
#   Output Params   :   df(Data Frame)
#   Description     :   Find missing values
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
############################################################################################
def findMissingValues(dFrame):
    print(dFrame.isnull().sum())

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
    sns.heatmap(dFrame.select_dtypes(include=['number']).corr(),annot=True,cmap="coolwarm")
    plt.title("Feature co-relation heatmap")
    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"CoRelationMatrix.png"))
###########################################################################################
#   Function        :   findFeaturesAndTarget
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
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
#   Date            :   30 Oct 2025
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
#   Date            :   28 Sep 2025
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
#   Date            :   30 Oct 2025
#####################################################################################################
def trainPipeline(pipeline,xTrain,yTrain):
    pipeline.fit(xTrain,yTrain)
    return pipeline
#####################################################################################################
#   Function name    :  testModelAndAccuracyCalculation
#   Input Params     :  pipeline,xTrain,yTrain data set,algorithmDetails(Dictionary)
#   Output           :  Return pipeline object
#   Description      :  Train a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
#####################################################################################################
def testModelAndAccuracyCalculation(model,x_Test,y_Test,algorithmDetails):
    """Predict the test results"""
    model_predicted=model.predict(x_Test)
    """Accuracy calculations"""
    model_Accuracy=accuracy_score(y_Test,model_predicted)
    """Calculate confusion Matrix """
    model_ConfusionMatrix=confusion_matrix(y_Test,model_predicted)
    """Model classification report"""
    model_Classification_Report=classification_report(y_Test,model_predicted)
    algorithmDetails["Accuracy Score"].append(model_Accuracy*100)
    algorithmDetails["Confusion Matrix"].append(model_ConfusionMatrix)
    algorithmDetails["Classification Report"].append(model_Classification_Report)

###########################################################################################
#   Function        :   train_and_evaluate
#   Input Params    :   None
#   Output Params   :   -
#   Description     :   Train and Evaluate model using KNN and Decision Tree classifier
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
############################################################################################
def train_and_evaluate():
    #Load CSV and pre-process data
    df=preProcessData()
     #Split data set
    x_Train,x_Test,y_Train,y_Test=spliDataSet(df)

    algorithmDetails={"Algorithm Name":[],"TrainedModel":[],"Accuracy Score":[],
                      "Confusion Matrix":[],"Classification Report":[]}
    
    """Creating pipelines for 2 models"""    
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
    #Plot and save accuracy plot and accuracy score
    plotAndSaveAccuracyScore(algorithmDetails)
    #Plot confusion matrix
    plotConfusionMatrix(algorithmDetails)
    #Save classification Report
    save_classification_report(algorithmDetails)
    
###########################################################################################
#   Function        :   plotIrisVariety
#   Input Params    :   df(Data Frame)
#   Output Params   :   Saves scatter plot of Iris Variety
#   Description     :   prints and saves scatter plot
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
##########################################################################################
def plotIrisVariety(df):
    ensure_dir(ARTIFACT_DIR)
    outDir=os.path.join(ARTIFACT_DIR,"featureWisePlot.png")

    fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(10,10))
    sns.scatterplot(x='sepal.length', y='sepal.width',
                hue='variety', data=df,ax=axes[0],palette="deep")
    sns.scatterplot(x='petal.length', y='petal.width',
                hue='variety', data=df,ax=axes[1],palette="deep")
    plt.savefig(outDir)
    plt.close()

###########################################################################################
#   Function        :   save_classification_report
#   Input Params    :   algorithmDetails(Dictionary holding all details)
#   Output Params   :   Saves classification report
#   Description     :   prints and saves classification report with precision,recall,F1 score
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
##########################################################################################
def save_classification_report(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    reportData=""
    for cnt in range(len(algorithmDetails["Classification Report"])):  
        reportData=f"{reportData}\t{algorithmDetails["Algorithm Name"][cnt]}\n{BORDER}" 
        clsReport=algorithmDetails["Classification Report"][cnt]
        reportData=f"{reportData}\n{clsReport}{BORDER}\n\n"
    with open (os.path.join(ARTIFACT_DIR,"classification_report.txt"),"w") as f:
        f.write(reportData)
###########################################################################################
#   Function        :   plotConfusionMatrix
#   Input Params    :   Dictionary holding all details
#   Output Params   :   -
#   Description     :   Saves confusion matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
############################################################################################    
def plotConfusionMatrix(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    display_labels = ["setosa", "versicolor","virginica"]
    """Create the figure and subplots
    As we used 2 algorithms use Subplots to plot 1 confusion matrix"""   
    fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(10,10))
    for cnt in range(len(algorithmDetails["Confusion Matrix"])):
         confuMatrix=ConfusionMatrixDisplay(
             confusion_matrix=algorithmDetails["Confusion Matrix"][cnt],
             display_labels=display_labels)   
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
#   Date            :   30 Oct 2025
############################################################################################    
def plotAndSaveAccuracyScore(algorithmDetails):
    ensure_dir(ARTIFACT_DIR)
    x=algorithmDetails['Algorithm Name']
    y=algorithmDetails['Accuracy Score']
    plt.figure(figsize=(8,7))
    plt.bar(x,y,width=0.4)
    plt.title('Accuracy of algorithms')
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")
    plt.yticks(y)
    plt.grid(True)
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
#   Date            :   30 Oct 2025
#####################################################################################################
def saveTrainedModel(model,modelName):    
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)
    print(f"Model saved to path :{path}") 
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
#####################################################################################################
def loadTrainedModel(modelName): 
    path=modelName+".joblib"
    path=os.path.join(ARTIFACT_DIR,path)
    model=joblib.load(path)  
    print(f"Model loaded from the path :{path}")
    return model    
###########################################################################################
#   Function        :   plotConfusionMatrix
#   Input Params    :   Number of samples
#   Output Params   :   -
#   Description     :   Test Model using test data
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
############################################################################################    
def test_models(n_samples):
    df=preProcessData()  
    sampleTestData =df.sample(n=n_samples) 
    resultMapping = {0: "setosa", 1: "versicolor",2:"virginica"}
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
        testResult=f"\n{testResult}{clf_Name}\n\
                        Testing Accuracy:{predictionAccuracy*100}\n{BORDER}\n"
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
#   Date            :   30 Oct 2025
#####################################################################################################
def testTrainedModel(model,sampleTestData): 
    predictedResult=model.predict(sampleTestData.drop(columns=[TARGET_COLUMN]))
    predictionAccuracy=accuracy_score(sampleTestData[TARGET_COLUMN],predictedResult)
    return predictionAccuracy,predictedResult

#####################################################################################################
#   Function Name    :  IrisClassification 
#   Description      :  Manages calls to training and testing functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
#####################################################################################################
def IrisClassification():
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
            "python3 IrisClassification.py --train\n"
            "python3 IrisClassification.py --test\n"
        )    
#####################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author          :   Vaishali M Jorwekar
#   Date            :   30 Oct 2025
#####################################################################################################
def main():
    IrisClassification()
#---------------------------------------------------------------------------------------------------------
# Main entry point of the application
#---------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    main()