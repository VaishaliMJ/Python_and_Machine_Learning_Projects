"""---------------------------------------------------------------------------------------------------------------
                Movie Recommendation System(Unsupervised -K Means)
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Built a recommendation engine using clustering & similarity measures
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import streamlit as st
import os,joblib
import pandas as pd
import numpy as np

ARTIFACT_DIR="artifact_MovieRecommondation"
MODEL_NAME="MovieRecommonder"
# --- Load Pre-calculated Data ---
DATA_FRAME_DICT="movies_dict"
COSINE_SIM = "cosine_sim_matrix"
API_KEY="93dbb41e595c1c945c2a61d97f692d61"
BASE_IMAGE_URL = 'https://image.tmdb.org/t/p/w500/' # w500 is a common size





import requests

def fetch_poster_url(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"



    try:
        response = requests.get(url)
        data = response.json()
        poster_path = data.get('poster_path')
        
        if poster_path:
            full_path = f"{BASE_IMAGE_URL}{poster_path}"
            return full_path
        else:
            return None # Or a placeholder image URL
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None


#####################################################################################################
#   Function name    :  loadModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
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
#   Description     :   Movie Recommender main function
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
#########################################################################################################
def main():
    st.title("Movie Recommendation")

    movies_dict=loadModel(DATA_FRAME_DICT)
    movies_df=pd.DataFrame(movies_dict)
    movies_list=movies_df['title'].values
    selectedMovieName=st.selectbox(label= " ",options=movies_list)

    if st.button("Recommend Movie"):
        recommended_movies,posters=recommend(movies_df,selectedMovieName)
        cols = st.columns(len(recommended_movies))
        for i, col in enumerate(cols):
            with col:
                st.image(posters[i], caption=recommended_movies[i]) 

#######################################################################################################
#   Function        :   recommend
#   Input Params    :   movies_df,movieName
#   Output Params   :   None
#   Description     :   Movie Recommender function
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
#########################################################################################################
def recommend(movies_df,movieName):
    movie_index=movies_df[movies_df['title']==movieName].index[0]
    
    # Load cosie similarity matrix
    Cosine_similarityMatrix=loadModel("cosine_sim_matrix")
   


    # Get the cluster ID of the selected movie
    movie_cluster = movies_df.iloc[movie_index]['Cluster']
    #st.write(f"Cluster:{movie_cluster}")
    
    # Get indices of all movies within the same cluster
    cluster_indices = movies_df[movies_df['Cluster'] == movie_cluster].index.tolist()

    movie_similarity=list(enumerate(Cosine_similarityMatrix[movie_index,cluster_indices]))
     

    movie_similarity=sorted(movie_similarity,reverse=True,key=lambda x : x[1])

    #st.write(movies_list)
    recommended_movies=[]
    recommended_posters=[]
    # Get the top 10 most similar movies (excluding the movie itself, which is 1st)
    top_movies_indices_relative = [i[0] for i in movie_similarity if i[0] != 0][:10]
    
    # Map relative cluster indices back to original DF indices
    final_recommendation_indices = [cluster_indices[i] for i in top_movies_indices_relative]
    
    for i in final_recommendation_indices:
            rec_movie_id = movies_df.iloc[i]['movie_id']
            poster_url = fetch_poster_url(rec_movie_id)
            if poster_url:
                recommended_movies.append(movies_df['title'].iloc[i])
                recommended_posters.append(poster_url)
            if len(recommended_movies) >= 10:
                break
       
    
    # Return the titles of the top 10 movies
    
    return  recommended_movies,recommended_posters  





#########################################################################################################
#   Starter
#########################################################################################################
if __name__=="__main__":
    main()       

