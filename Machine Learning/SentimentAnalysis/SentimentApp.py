"""---------------------------------------------------------------------------------------------------------------
                    Sentiment Analysis App
            Sentiment Analysis Enhancement (Ensemble Bagging & Boosting)
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Sentiment Analysis Enhancement (Random Forest and Gradient Boosting)
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import os,joblib,re
import streamlit as st
from nltk.stem.porter import PorterStemmer

#####################################################################################################
#   Constants and file names
#####################################################################################################
BORDER="-"*65
TWEETS_DATA="Tweets.csv"
ARTIFACT_DIR="artifact_sentimentAnalysis"
RF_MODEL_NAME="Random Forest Classifier"
GB_MODEL_NAME="Gradient Boosting Classifier"
TFIDF_MATRIX="tfidf_matrix"
VOTING_CLASSIFIER="Voting Classifier"
###########################################################################################
#   Function        :   processAndStemText
#   Input Params    :   text
#   Output Params   :   None
#   Description     :   pre-process text column
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
############################################################################################
def processAndStemText(text):
    text=text.lower()
    #   Alphabets only
    text = re.sub(r"[^a-zA-Z\s]", "", text)  
    #   remove https ,www,@ 
    text=re.sub(r'http\S+|www\S+',"",text)
    #   Remove extra white spaces
    text = re.sub(r"\s+", " ", text).strip() 
    
    stemWords=[]
    ps=PorterStemmer()
    for word in text.split():
        stemWords.append(ps.stem(word))
    return " ".join(stemWords)  
#####################################################################################################
#   Function name    :  loadModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
#####################################################################################################
def loadModel(modelName):  
    path=modelName+".joblib"
    path=os.path.join(ARTIFACT_DIR,path)
    model = joblib.load(path)
    print(f"Model loaded from the path :{path}")
    return model  

#####################################################################################################
#   Function name    :  loadModel
#   Input Params     :  sentimentPredicted
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   19 Dec 2025
#####################################################################################################
def analyseSentiment(sentimentPredicted): 
    if sentimentPredicted == 2 :
        label="Positive 😊" 
        color="green"
    elif sentimentPredicted==0 :
        label="Negative 😔" 
        color="red"
    else:
        label="Neutral 😑"
        color="grey"  
    return label,color     
#########################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  19 Dec 2025
#########################################################################################################
def main():
    st.title("🚀 Text Sentiment Analyzer(😊😔😑)")
    #   Load Model
    rfModel=loadModel(RF_MODEL_NAME)   
    gbModel=loadModel(GB_MODEL_NAME)   
    tfidf=loadModel(TFIDF_MATRIX)   
    vcModel=loadModel(VOTING_CLASSIFIER)
   
        
    user_input=st.text_area("Enter text to analyse sentiment",value="")
    cleaned_input = processAndStemText(user_input)
    vectorized_input = tfidf.transform([cleaned_input])
    if st.button("🚀 Predict Sentiment"):
        rf_sentiment=rfModel.predict(vectorized_input)[0]
        gb_sentiment=gbModel.predict(vectorized_input)[0]
        vc_sentiment=vcModel.predict(vectorized_input)[0]
        #color = ['green','grey','red']
        label = ""
        color=""
        col1, col2,col3 = st.columns(3)
        #Neutral : 1. negative :0, positive : 2
        with col1:
            st.write("**Random Forest**")   
            label,color=analyseSentiment(rf_sentiment)
            st.write(f"** :{color}[{label}]")
            
        with col2:
            st.write("**Gradient Boosting**")
            label,color=analyseSentiment(gb_sentiment)
            st.write(f"** :{color}[{label}]")
        with col3:
            st.write("**Voting Classifier**")
            label,color=analyseSentiment(vc_sentiment)
            st.write(f"** :{color}[{label}]")    
            
        st.info(f"**Processed Text (Stemmed):** {cleaned_input}")
    else:
        st.warning("Please enter some text first.")
    st.sidebar.markdown("### Model Details\n")
    st.sidebar.write("- **Vectorization:** TF-IDF")
    st.sidebar.write("- **Preprocessing:** Stemming & Cleaning")
    st.sidebar.write(f"- **Models:**\n Random Forest(Bagging)\n,Gradient Boosting \n and Voting Classifer")    
#########################################################################################################
#   Starter
#########################################################################################################
if __name__=="__main__":
    main()