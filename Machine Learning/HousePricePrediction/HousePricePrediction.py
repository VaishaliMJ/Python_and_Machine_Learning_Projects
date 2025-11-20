"""-----------------------------------------------------------------------------------------------------
                         House Price Prediction
                    (Student name - Vaishali Jorwekar)
                    Python : Marvellous Infosystems
--------------------------------------------------------------------------------------------------------
Problem statement: Regression based project to predict property prices using multiple features
--------------------------------------------------------------------------------------------------------"""
#####################################################################################################
# Required Python Packages
#####################################################################################################
import os,argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

#####################################################################################################
#   Constants
#####################################################################################################
ARTIFACT_DIR="artifact_HousePrice"
DATASET_FILENAME="House Price India.csv"
BORDER="-"*65
TARGET_COLNAME="Price"
TEST_SIZE=0.2
RANDOM_STATE=42
#Model name
CLF_MODELS={"Linear Regression": LinearRegression()}
#####################################################################################################
#   Function Name   :   ensure_dir
#   Input Params    :   path of directory
#   Output Params   :   None
#   Description     :   Check and create ARTIFACT_DIR if does not exists
#   Author          :   Vaishali M. Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def ensure_dir(path:str):
    os.makedirs(ARTIFACT_DIR,exist_ok=True)
#####################################################################################################
#   Function Name   :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,testing baselines
#   Author          :   Vaishali M. Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def parse_args():
    p=argparse.ArgumentParser(description="House Price Prediction Case Study")
    p.add_argument("--train",action="store_true",
                   help="Training using regression models")
    p.add_argument("--test",action="store_true",
                   help="Testing of pre-trained models")
    p.add_argument("--samples",type=int,default=10,help="Number of samples for testing ")
    return p.parse_args()
###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
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
#   Function        :   displayCoRelationMatrix
#   Input Params    :   dFrame(data frame)
#   Output Params   :   -
#   Description     :   Displays co-relation matrix
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
############################################################################################
def displayCoRelationMatrix(dFrame):
    print(BORDER)
    ensure_dir(ARTIFACT_DIR)
    plt.figure(figsize=(18,10))
    
    sns.heatmap(dFrame.corr(),annot=True,cmap="coolwarm")
    plt.title("Feature co-relation heatmap")
    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"CoRelationMatrix.png"))
#####################################################################################################
#   Function Name   :   preProcessData
#   Input Params    :   Directory Name
#   Output Params   :   None
#   Description     :   Train and evaluate the models
#   Author          :   Vaishali M. Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def preProcessData():
    df=pd.DataFrame()
    try:
        #Load data from CSV file
        df=readCSVFile(csvFileName=DATASET_FILENAME)
       
    except FileNotFoundError:
        print(f"File : {DATASET_FILENAME} not found !!!")   
        exit()
    #   Clean dataset
    df=cleanDataSet(df)
    #   Co-Relation matrix 
    displayCoRelationMatrix(df)
    #Encode data set
    #df=applyOneHotEncoding(df)
    return df
#####################################################################################################
#   Function Name   :   applyOneHotEncoding
#   Description     :   Check and handle categorical data columns
#   Input params    :   dFrame(Data frame set)
#   Output          :   Handle Categorical data
#   Author          :   Vaishali M. Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def applyOneHotEncoding(dFrame):
    ensure_dir(ARTIFACT_DIR)
    object_columns = dFrame.select_dtypes(include='object')
    print(object_columns)
    OHE=OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform="pandas")
    for colName in dFrame.select_dtypes(include=['object']).columns: 
        OHETransformed=OHE.fit_transform(dFrame[[colName]])   
        dFrame=pd.concat([dFrame,OHETransformed],axis=1).drop(columns=[colName])
    print(dFrame.head())   
    dFrame.to_csv(os.path.join(ARTIFACT_DIR,"OHEData.csv"))
    return dFrame   
###########################################################################################
#   Function        :   cleanDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   Updated data frame 
#   Description     :   Cleans data set by removing unwanted columns
#   Author          :   Vaishali M Jorwekar
#   Date            :   15 Nov 2025
############################################################################################
def cleanDataSet(dFrame)->pd.DataFrame:
    print(dFrame.info())

    print(dFrame.describe())
    
    print(BORDER)
    print("Cleaning Data set....")
    print(BORDER)
    print("Dimensions of data set before cleaning is :",dFrame.shape)
    print(BORDER)

    print("Checking for 'na' values...")
    print(BORDER)
  
    print(dFrame.isna().sum())
    print(BORDER)

    print("Duplicated values...")
    print(dFrame.duplicated().sum())

    # Replace Target column empty values with column mean() values
    dFrame[TARGET_COLNAME]=dFrame[TARGET_COLNAME].fillna(dFrame[TARGET_COLNAME].mean())
    dFrame.dropna(inplace=True)
    print("Checking for null values...")
    print(dFrame.isnull().sum())
    return dFrame


###########################################################################################
#   Function        :   findFeatures
#   Input Params    :   dFrame(data frame)
#   Output Params   :   Feature Extraction
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
############################################################################################
def findFeatures(dFrame):
    xFeatures=dFrame[['number of bedrooms', 'number of bathrooms','living area', 
                      'grade of the house','Area of the house(excluding basement)',
                      'living_area_renov']]
    return xFeatures
###########################################################################################
#   Function        :   findFeaturesAndTarget
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
############################################################################################
def findFeaturesAndTarget(dFrame):
    print("dFrame Columns:\n",dFrame.columns)
    xFeatures=findFeatures(dFrame)
    #xFeatures=dFrame.drop(columns=[TARGET_COLNAME])
    
    yTarget=dFrame[TARGET_COLNAME]
    print(BORDER)    
    return xFeatures,yTarget
###########################################################################################
#   Function        :   spliDataSet
#   Input Params    :   dFrame(data frame)
#   Output Params   :   independent and dependent variables
#   Description     :   This method spilts data set into features and labels
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
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
#   Input Params     :  clf_Model : model name
#   Output           :  Return pipeline object
#   Description      :  Build a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def build_Pipeline(clf_Model):  
    pipe=Pipeline(steps=[
        ("scalar",StandardScaler()),
        ("clf",clf_Model)

    ]) 
    return pipe
#####################################################################################################
#   Function name    :  trainPipeline
#   Input Params     :  pipeline,xTrain,yTrain data set
#   Output           :  Return pipeline object
#   Description      :  Train a pipeline
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def trainPipeline(pipeline,xTrain,yTrain):
    pipeline.fit(xTrain,yTrain)
    return pipeline 
###########################################################################################
#   Function        :   plotPredictedPricesVsActual
#   Input Params    :   y_Test(Actual House Prices),y_Predicted(Predicted House Prices)
#   Output Params   :   -
#   Description     :   Plots the Actual Vs Predicted Prices Plot
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
############################################################################################ 
def plotPredictedPricesVsActual(y_Test,predictedHousePrices):
    plt.figure(figsize=(10,8))
    plt.scatter(y_Test,predictedHousePrices,color="blue",label="Actual")
    min_val = min(min(y_Test), min(predictedHousePrices))
    max_val = max(max(y_Test), max(predictedHousePrices))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--') # 'r--' creates a red dashed line
    plt.xlabel("Actual House Prices")
    plt.ylabel("Predicted House Prices")
    plt.title("Actual Vs Predicted House Prices Plot")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(os.path.join(ARTIFACT_DIR,"HousePricesRegressionPlot.png"))   
###########################################################################################
#   Function        :   result_PredictionAndError
#   Input Params    :   model,x_Tes(features),y_Test(Target)
#   Output Params   :   -
#   Description     :   Predicts Results and calculates errors
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
############################################################################################
def result_PredictionAndError(model,x_Test,y_Test):
    yPredict=model.predict(x_Test)
    mse=mean_squared_error(y_Test,yPredict)
    rmse=np.sqrt(mse)
    r2=r2_score(y_Test,yPredict)

    errorDetails=f"{BORDER}\nMean Square and R^2 Details:\n{BORDER}\n"
    errorDetails=f"{errorDetails}Mean squared error:{mse}\nRoot mean squared error:{rmse}\nR square value:{r2}"
   
    outDir=os.path.join(ARTIFACT_DIR,"MeanSquaredError.txt")
    with open(outDir,"w") as fErr:
         fErr.write(str(errorDetails))
    return yPredict     
#####################################################################################################
#   Function name    :  saveTrainedModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def saveTrainedModel(model,modelName):
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)
    print(f"Model saved to path :{path}")    
#####################################################################################################
#   Function Name   :   trainAndEvaluate
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Train and evaluate the models
#   Author          :   Vaishali M. Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def trainAndEvaluate():
    #   Pre-Process data
    df=preProcessData()
    #   Split data set
    x_Train,x_Test,y_Train,y_Test=spliDataSet(df)
    for clf_Name,clfModel in CLF_MODELS.items():
        # Build and train pipeline
        pipeline=build_Pipeline(clfModel)
        # Train Pipeline
        trainedModel=trainPipeline(pipeline,x_Train,y_Train)
        predicteHousePrice=result_PredictionAndError(trainedModel,x_Test,y_Test)
        plotPredictedPricesVsActual(y_Test,predicteHousePrice)
        # Save trained model
        saveTrainedModel(trainedModel,clf_Name)
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
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
#   Date            :   5 Nov 2025
#####################################################################################################
def testTrainedModel(model,sampleTestData): 
     xFeatures=findFeatures(sampleTestData)
     predictedResult=model.predict(xFeatures)
     np.set_printoptions(precision=2)
     return predictedResult           
###########################################################################################
#   Function        :   test_Model
#   Input Params    :   n_samples(Number of samples)
#   Output Params   :   -
#   Description     :   Test the model
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
############################################################################################    
def test_Model(n_samples):
    df=preProcessData()
    sampleTestData =df.sample(n=n_samples)  
    #Copy test data
    copyTestData=sampleTestData.copy()
    print(BORDER)
    print(f"Selected sample data for testing are \n: {sampleTestData}")
    print(BORDER)
    for clf_Name,clfModel in CLF_MODELS.items():
        print(f"Sample testing Using :'{clf_Name}'") 
        # Load the model and test samples
        trainedModel=loadTrainedModel(clf_Name)
        predicteHousePrices=testTrainedModel(trainedModel,sampleTestData)
        # Save testing results
        save_Test_Results(copyTestData,predicteHousePrices)
#####################################################################################################
#   Function name    :  save_Test_Results
#   Input Params     :  dataFrame,PredictedResults
#   Output           :  model
#   Description      :  Test the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def save_Test_Results(dataFrame,PredictedResults):
    dataFrame["Predicted House Prices"]=PredictedResults
    outDir=os.path.join(ARTIFACT_DIR,"PredictedHousePrices.txt")
    with open(outDir,"w") as fErr:
         fErr.write(str(dataFrame))        
#####################################################################################################
#   Function Name   :   housePricePrediction
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main function of the program
#   Author          :   Vaishali M. Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def housePricePrediction():
    ensure_dir(ARTIFACT_DIR)
    did_anything=False
    args=parse_args()
    if args.train:
        trainAndEvaluate()
        did_anything=True
    if args.test:
        test_Model(n_samples=args.samples)
        did_anything=True
    if not did_anything:
        print(
            "Nothing to do .Try one of the :\n"
            "python3 HousePricePrediction.py --train\n"
            "python3 HousePricePrediction.py--test\n"
        )               
#####################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def main():
    housePricePrediction()
#####################################################################################################
#   Starter
#####################################################################################################
if __name__=="__main__":
    main()

