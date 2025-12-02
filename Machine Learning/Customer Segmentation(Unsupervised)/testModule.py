"""---------------------------------------------------------------------------------------------------------------
                       Customer Segmentation(Unsupervised -K Means clustering)   
                        Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Customer Segmentation(Unsupervised -K Means clustering)-
                Grouped retail customers into distinct clusters based on purchasing behaviour for targeted marketing
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports and constants
#####################################################################################################
import streamlit as st
import os,joblib
import pandas as pd
ARTIFACT_DIR="artifact_CustomerSegmentation"
MODEL_NAME="CustomerSegmentation"
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
#####################################################################################################
def loadTrainedModel(modelName):  
    path=modelName+".joblib"
    path=os.path.join(ARTIFACT_DIR,path)
    model = joblib.load(path)
    print(f"Model loaded from the path :{path}")
    return model      

###########################################################################################
#   Function        :   testModel
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Test pre-trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   28 Nov 2025
############################################################################################
def testModel():   
    #   Load Model
    kmeans=loadTrainedModel(MODEL_NAME)   
    #   Scalar loading
    scalar=loadTrainedModel("scalar")
    #   App
    st.title("Customer Segmentation Testing App")
    st.write("Enter Customer Details to predict segment")
    featureColNames=["Age","Spending","Income","NumWebPurchases","NumStorePurchases","NumWebVisitsMonth","Recency"]

    #   Customer details
    custAge=st.number_input("Age",min_value=18,max_value=100,value=18)
    custIncome=st.number_input("Income",min_value=0,max_value=200000,value=1000)
    totalSpending=st.number_input("Spending(Total num of purchases)",min_value=0,max_value=200000,value=2000)
    numWebPurchases=st.number_input("Number of web purchases",min_value=0,max_value=200,value=20)
    numStorePurchases=st.number_input("Number of store purchases",min_value=0,max_value=200,value=10)
    numWebVisits=st.number_input("Number of web visits",min_value=0,max_value=200,value=20)
    recency=st.number_input("Recency(Days since last purchase)",min_value=0,max_value=365,value=5)

    user_data=pd.DataFrame({
            "Age":[custAge],
            "Spending":[totalSpending],
            "Income":[custIncome],
            "NumWebPurchases":[numWebPurchases],
            "NumStorePurchases":[numStorePurchases],
            "NumWebVisitsMonth":[numWebVisits],
            "Recency":[recency]
    })

    inputScaledData=scalar.transform(user_data)

    if st.button("Predict Segment"):
        cluster=kmeans.predict(inputScaledData)[0]
        st.success(f"Predicted Customer Segment : Cluster {cluster}")
#########################################################################################################
#   Starter
#########################################################################################################
if __name__=="__main__":
    testModel()        