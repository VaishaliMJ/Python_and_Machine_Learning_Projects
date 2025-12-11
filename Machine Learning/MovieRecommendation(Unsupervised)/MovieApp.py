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
import os,joblib,json
import pandas as pd
import numpy as np
from dotenv import load_dotenv 

EXTRACTED_DATA_FRAME="ExtractedDataFrame"
ARTIFACT_DIR="artifact_MovieRecommondation"
MODEL_NAME="MovieRecommonder"
# --- Load Pre-calculated Data ---
DATA_FRAME_DICT="movies_dict"
COSINE_SIM = "cosine_sim_matrix"
#load_dotenv()  
config=json.load(open("config.json"))
    
API_KEY=config["API_KEY"]
BASE_IMAGE_URL = 'https://image.tmdb.org/t/p/w500/'



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
    # loading variables from .env file
    
    st.title(" 🎬 Movie Recommendation")

    movies_dict=loadModel(DATA_FRAME_DICT)
    movies_df=pd.DataFrame(movies_dict)
    movies_list=movies_df['title'].values
    selectedMovieName=st.selectbox(label= " ",options=movies_list)
    #movie_index=movies_df[movies_df['title']==selectedMovieName]
    Extracted_movies_dict=loadModel(EXTRACTED_DATA_FRAME)
    originalDF=pd.DataFrame(Extracted_movies_dict)
    movie_index=movies_df[movies_df['title']==selectedMovieName].index[0]

    if st.button(" 🚀 Recommend Similar Movies with titles only"):
        recommended_movies,posters,recommended_movie_ids=recommend(movies_df,selectedMovieName)

        cols = st.columns(len(recommended_movies))
        for i, col in enumerate(cols):
            with col:
                st.image(posters[i], caption=recommended_movies[i]) 
    if st.button(" 🚀 Recommend Similar Movies with overview"):
        recommended_movies,posters,recommended_movie_ids=recommend(movies_df,selectedMovieName)
        #print(recommended_movies)
        
        #movieLen = len(recommended_movie_ids)
        #for i, col in enumerate(cols):
        #    with col:
        #       st.image(posters[i], caption=recommended_movies[i])

        for i in range(len(recommended_movie_ids)):

            with st.container():
                col1,col2=st.columns([1,3])   
            
                with col1: 
                   st.image(posters[i],caption=recommended_movies[i],width=100) 
                with col2:
                    
                    movieId = originalDF[originalDF['movie_id'] == recommended_movie_ids[i]].index[0]
                    #movie_Overview = originalDF.iloc[recommended_movie_ids[i]]['overview'].index[0]

                    movie_Overview = originalDF.iloc[movieId]['overview']

                    #movie_Overview = originalDF.iloc[i]['overview']
                    
                    st.write(f"Overview:")
                    st.write(movie_Overview)
                    st.divider()
    #if not movie_index.empty:
       
    if st.checkbox('Show Movie Details'):

        showMovieDetails(movie_index,originalDF)
        
        
    st.divider()
#######################################################################################################
#   Function        :   showMovieDetails
#   Input Params    :   dataFrame,originalDF
#   Output Params   :   None
#   Description     :   Movie Recommender main function
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
#########################################################################################################
def showMovieDetails(movie_index,originalDF):

    #Extracted_movies_dict=loadModel(EXTRACTED_DATA_FRAME)
    #originalDF=pd.DataFrame(Extracted_movies_dict)
    movie_id = originalDF.iloc[movie_index]['movie_id']
    poster_url = fetch_poster_url(movie_id)
    movie_Title = originalDF.iloc[movie_index]['title']
    movie_Overview = originalDF.iloc[movie_index]['overview']
    movie_Cast_list = originalDF.iloc[movie_index]['cast']
    movie_genres=originalDF.iloc[movie_index]['genres']
    movie_Crew_list = originalDF.iloc[movie_index]['crew']
    movie_Cast=""
    for i, item in enumerate(movie_Cast_list):
        movie_Cast += item
        movie_Cast += " "
    with st.container():
        col1,col2,col3=st.columns([1,2,4])   
        with col1 :  
            st.image(poster_url,caption=movie_Title,width=100) 
        with col2:
            st.write("**Title**:")
            st.write(movie_Title)
            st.write("**Genres**:")
            st.markdown("\t ,".join([f" {genre}" for genre in movie_genres]))
            st.write("**Overview**:")
            st.write(movie_Overview)
        with col3:    
            st.write("**Director**:")
            st.markdown(" ".join([f"* {actor}" for actor in movie_Crew_list]))
            st.write("**Cast Information**")
            st.markdown("\n".join([f"* {actor}" for actor in movie_Cast_list]))
            
            displayMovieCast(movie_id,movie_Title)
    
#######################################################################################################
#   Function        :   fetchMovieCastDetails
#   Input Params    :   movie_index
#   Output Params   :   None
#   Description     :   Movie cast fetch function
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
#########################################################################################################
def fetchMovieCastDetails(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}&language=en-US"
    #print(url)
    try:
        response = requests.get(url)
        data = response.json()
        cast_list = data.get('cast', [])
        #print(data)
        cast_details = []
        # Get top 5 cast members
        for person in cast_list[:5]:
            person_name = person.get('name')
            profile_path = person.get('profile_path')
            if person_name and profile_path:
                # Construct the full image URL directly
                full_path = f"https://image.tmdb.org/t/p/w185/{profile_path}"
                cast_details.append({'name': person_name, 'poster_url': full_path})
        return cast_details

    except requests.exceptions.RequestException:
        return []    

#######################################################################################################
#   Function        :   displayMovieCast
#   Input Params    :   movies_df,movieName
#   Output Params   :   None
#   Description     :   Movie Recommender function
#   Author          :   Vaishali M Jorwekar
#   Date            :   4 Dec 2025
#########################################################################################################
def displayMovieCast(movie_id,movieName):
    cast_details = fetchMovieCastDetails(movie_id)
    st.subheader(f"Top Cast for : {movieName} ")
    if cast_details:
        cast_cols = st.columns(len(cast_details))
        for i, cast_member in enumerate(cast_details):
            with cast_cols[i]:
                st.image(cast_member['poster_url'], caption=cast_member['name'])
    else:
        st.info("Cast details could not be retrieved.")

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
    recommended_movies_title=[]
    recommended_movie_ids=[]
    recommended_posters=[]
    # Get the top 10 most similar movies (excluding the movie itself, which is 1st)
    top_movies_indices_relative = [i[0] for i in movie_similarity if i[0] != 0][:10]
    #top_movies_indices_relative = [i[0] for i in movie_similarity][:11]

    # Map relative cluster indices back to original DF indices
    final_recommendation_indices = [cluster_indices[i] for i in top_movies_indices_relative]
    
    for i in final_recommendation_indices:
            rec_movie_id = movies_df.iloc[i]['movie_id']
            poster_url = fetch_poster_url(rec_movie_id)
            if poster_url:
                recommended_movies_title.append(movies_df['title'].iloc[i])
                recommended_posters.append(poster_url)
                recommended_movie_ids.append(rec_movie_id)
            if len(recommended_movies_title) >= 10:
                break
       
    
    # Return the titles of the top 10 movies
    
    return  recommended_movies_title,recommended_posters,recommended_movie_ids



#########################################################################################################
#   Starter
#########################################################################################################
if __name__=="__main__":
    main()       

