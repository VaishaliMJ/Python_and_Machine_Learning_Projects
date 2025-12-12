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

import streamlit as st
import os,joblib
import numpy as np
ARTIFACT_DIR="artifact_HousePrice"


#####################################################################################################
#   Function name    :  loadModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
#####################################################################################################
def loadModel(modelName):  
    path=modelName+".joblib"
    path=os.path.join(ARTIFACT_DIR,path)
    model = joblib.load(path)
    print(f"Model loaded from the path :{path}")
    return model      

#######################################################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Price Predition main function
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
#########################################################################################################
def main():
    #xFeatures=dFrame[['number of bedrooms', 'number of bathrooms','living area', 
            #         'condition of the house','Number of schools nearby']]
    st.title("House Price Prediction")
    st.divider()
    st.write("This App predicts house price.Enter below details to test App")
    st.divider()
    bedrooms=st.number_input("Number of bedrooms",min_value=0,value=0)
    bathrooms=st.number_input("Number of bathrooms",min_value=0,value=0)
    livingArea=st.number_input("Living Area",min_value=0,value=2000)
    houseCondition=st.number_input("Condition of house",min_value=0,value=3)
    schoolsNearby=st.number_input("Number of Schools nearby",min_value=0,value=0)
    st.divider()

    model=loadModel("Linear Regression")
    X=[[bedrooms,bathrooms,livingArea,houseCondition,schoolsNearby]]
    predictButton=st.button("Predict House Price")
    if predictButton:
        X_values=np.array(X)
        predictedValue=model.predict(X_values)[0]
        st.write(f"Predicted Price:{predictedValue:,.2f}")

    else:
        st.write("Use Predict button to predcit values")    

#########################################################################################################
#   Starter
#########################################################################################################
if __name__=="__main__":
    main()      