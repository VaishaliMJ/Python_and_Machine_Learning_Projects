"""-----------------------------------------------------------------------------------------------------
                     Adverstising Sales Analysis
                    Student name - Vaishali Jorwekar
-----------------------------------------------------------------------------------------------------
Problem statement: Built Regression model to analyse the sales of an advertising agenecy 
-----------------------------------------------------------------------------------------------------"""
####################################################################################################
# Required Python Packages
####################################################################################################
import os,argparse,joblib
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

####################################################################################################
# Constants and file names
####################################################################################################
BORDER="-"*65
DATASET_FILENAME="Advertising.csv"
ARTIFACT_DIR="artifact_AdSalesAnalysis"
TEST_SIZE=0.2
RANDOM_STATE=42
TARGET_COLUMN="sales"
MODEL_NAME="Linear Regression"
###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,testing
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
############################################################################################
def parse_args():
    p=argparse.ArgumentParser(description="Advertising Sale Analysis")
    p.add_argument("--train",action="store_true",help="Train using Linear Regression") 
    p.add_argument("--test",action="store_true",help="Test pre-trained models")
    p.add_argument("--samples",type=int,default=10,help="Number of samples for testing")
    return p.parse_args()  
###########################################################################################
#   Function        :   train_and_evaluate
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Train and Evaluate the model
#   Date            :   25 Nov 2025
############################################################################################
def train_and_evaluate():
    #Load CSV and pre-process data
    df=preProcessData()
    #Split data set  
    x_Train,x_Test,y_Train,y_Test=spliDataSet(df)
     #Linear Regression
    LinReg=build_Pipeline(LinearRegression())
    #Train Model
    LinReg=trainPipeline(LinReg,x_Train,y_Train)
    # Test model
    predicteSales=result_PredictionAndError(LinReg,x_Test,y_Test)
    #   Actual Vs Predicted Values
    plotSales(x_Test,y_Test,predicteSales)
    # Save trained model
    saveTrainedModel(LinReg,MODEL_NAME)
#####################################################################################################
#   Function name    :  saveTrainedModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
#####################################################################################################
def saveTrainedModel(model,modelName):
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)
    print(f"Model saved to path :{path}")        
###########################################################################################
#   Function        :   plotSales
#   Input Params    :   x_Test,y_Test,y_Predicted(Predicted Sales)
#   Output Params   :   -
#   Description     :   Plots the Actual Vs Predicted Sales values
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
############################################################################################ 
def plotSales(x_Test,y_Test,predicteSales):
    plt.figure(figsize=(10,8))

    plt.scatter(y_Test,predicteSales,color="blue")

    plt.plot([min(y_Test), max(y_Test)], [min(y_Test), max(y_Test)], 'r--', label='Regression Line') 
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual Vs Predicted Sales Regression Plot")
    #plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"SalesPredictionPlot.png"))

###########################################################################################
#   Function        :   result_PredictionAndError
#   Input Params    :   model,x_Tes(features),y_Test(Target)
#   Output Params   :   -
#   Description     :   Predicts Results and calculates errors
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
############################################################################################
def result_PredictionAndError(model,x_Test,y_Test):
    y_predict=model.predict(x_Test)

    mse=mean_squared_error(y_Test,y_predict)
    rmse=np.sqrt(mse)
    r2=r2_score(y_Test,y_predict)
    
    errorDetails=f"{BORDER}\nMean Square and R^2 Details:\n{BORDER}\n"
    errorDetails=f"{errorDetails}Mean squared error:{mse}\nRoot mean squared error:{rmse}\nR square value:{r2}"
   
    outDir=os.path.join(ARTIFACT_DIR,"MeanSquaredError.txt")
    with open(outDir,"w") as fErr:
         fErr.write(str(errorDetails))
    return y_predict         
#####################################################################################################
#   Function name    :  trainPipeline
#   Input Params     :  pipeline,xTrain,yTrain data set
#   Output           :  Return pipeline object
#   Description      :  Train a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
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
#   Date            :   25 Nov 2025
#####################################################################################################
def build_Pipeline(clf_Name):
    pipe=Pipeline(steps=[
         ("scalar",StandardScaler()),
         ("clf",clf_Name),
    ])
    return pipe      
###########################################################################################
#   Function        :   spliDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
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
#   Date            :   25 Nov 2025
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
#   Date            :   25 Nov 2025
############################################################################################
def preProcessData(csvFileName=DATASET_FILENAME)->pd.DataFrame:
    #Load data from CSV file
    df=readCSVFile(csvFileName)
    
    #Clean data set
    df=cleanDataSet(df)
    #Print co-relation matrix
    displayCoRelationMatrix(df)
    # Pair plot
    plotPairPlot(df)
    return df
###########################################################################################
#   Function        :   displayCoRelationMatrix
#   Input Params    :   dFrame(data frame)
#   Output Params   :   -
#   Description     :   Displays co-relation matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
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
#   Function        :   plotPairPlot
#   Input Params    :   dFrame(data frame)
#   Output Params   :   -
#   Description     :   Displays pair plot
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
############################################################################################
def plotPairPlot(dFrame):
    sns.pairplot(dFrame,hue=TARGET_COLUMN)
    #plt.show()
    plt.title("Pair plot")
    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"pairPlot.png"))   
  
###########################################################################################
#   Function        :   cleanDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   Updated data frame 
#   Description     :   Cleans data set by removing unwanted columns
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
############################################################################################
def cleanDataSet(dFrame)->pd.DataFrame:
    print(BORDER)
    print("Cleaning Data set....")
    print(BORDER)
    print("Dimensions of data set before cleaning is :",dFrame.shape) 
    # Drop column ['']
    dFrame.drop(['Unnamed: 0'],axis=1,inplace=True)
    dFrame.dropna(inplace=True)
    print(f"Checking for null values:\n{dFrame.isnull().sum()}")
    print(BORDER)
    print("Data set after cleaning:",dFrame.shape)
    print(dFrame.head())
    return dFrame   
###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov  2025
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
#   Function        :   test_Model
#   Input Params    :   n_samples(Number of samples)
#   Output Params   :   -
#   Description     :   Test the model
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
############################################################################################    
def test_Model(n_samples):
    df=preProcessData()
    sampleTestData=df.sample(n=n_samples)
    #Copy test data
    copyTestData=sampleTestData.copy()
    print(BORDER)
    print(f"Selected sample data for testing are \n: {sampleTestData}")
    print(BORDER)
    print(f"Sample testing Using :'{MODEL_NAME}'")  
    # Load the model and test samples
    trainedModel=loadTrainedModel(MODEL_NAME)
    predicteSales=testTrainedModel(trainedModel,sampleTestData)
    # Save testing results
    save_Test_Results(copyTestData,predicteSales)
#####################################################################################################
#   Function name    :  save_Test_Results
#   Input Params     :  dataFrame,PredictedResults
#   Output           :  model
#   Description      :  Test the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
#####################################################################################################
def save_Test_Results(dataFrame,PredictedResults):
    dataFrame["Predicted Sales"]=PredictedResults
    outDir=os.path.join(ARTIFACT_DIR,"PredictedSales.txt")
    with open(outDir,"w") as fErr:
         fErr.write(str(dataFrame))
        
#####################################################################################################
#   Function name    :  testTrainedModel
#   Input Params     :  path = MODEL_PATH,n_samples
#   Output           :  model
#   Description      :  Test the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
#####################################################################################################
def testTrainedModel(model,sampleTestData): 
     predictedResult=model.predict(sampleTestData.drop(columns=TARGET_COLUMN))
     return predictedResult          
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Nov 2025
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
#   Date            :   25 Nov 2025
#####################################################################################################
def main():
    ensure_dir(ARTIFACT_DIR)
    did_anything=False
    args=parse_args()
    if args.train:
        train_and_evaluate()
        did_anything=True
    if args.test:
        test_Model(n_samples=args.samples)
        did_anything=True
    if not did_anything:
        print(
            "Nothing to do .Try one of the :\n"
            "python3 AdvertisingSalesAnalysis.py --train\n"
            "python3 AdvertisingSalesAnalysis.py--test\n"
        )   
#####################################################################################################
#     Starter
#####################################################################################################
if __name__=="__main__":
    main()
