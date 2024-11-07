import requests
import pandas as pd
import streamlit as st
from groq import Groq

# Set your TMDb and Groq API keys
tmdb_api_key = ""
groq_api_key = ""
client = Groq(api_key=groq_api_key) 

# Function to fetch movies by genre from TMDb
def fetch_movies_by_genre(genre_id):
    url = f"https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": tmdb_api_key,
        "with_genres": genre_id,
        "sort_by": "popularity.desc"
    }
    response = requests.get(url, params=params)
    return response.json()['results'] if response.status_code == 200 else []

# Function to fetch available genres from TMDb
def fetch_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": tmdb_api_key}
    response = requests.get(url, params=params)
    return response.json()['genres'] if response.status_code == 200 else []


# Function to fetch movie poster URL from TMDb by title
def fetch_movie_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": tmdb_api_key,
        "query": title
    }
    response = requests.get(url, params=params).json()
    if response['results']:
        poster_path = response['results'][0].get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w200{poster_path}"
        else:
            return "https://plus.unsplash.com/premium_photo-1682125795272-4b81d19ea386?q=80&w=2060&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        
    return "https://plus.unsplash.com/premium_photo-1682125795272-4b81d19ea386?q=80&w=2060&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

# Streamlit Interface
st.set_page_config(page_title="CineMatch", page_icon=":popcorn:")
st.markdown("<h2 style='text-align: center;'>CineMatch 🤖</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Movie Recommender System </h3>", unsafe_allow_html=True)

# Create a single row for dropdowns and button side by side
col1, col2 = st.columns([1, 1])  # Set columns ratio (1:1:1)

with col1: 
    genres = fetch_genres()
    genre_options = {genre['name']: genre['id'] for genre in genres}
    selected_genre_name = st.selectbox("Select Genre", list(genre_options.keys()), key="genre", help="Choose a genre", index=0)
    selected_genre_id = genre_options[selected_genre_name]

with col2:  
    sort_by = st.selectbox("Sort by", ["Popularity", "Rating"], key="sort", help="Select sorting option")

# Trigger button
if st.button("Get Recommendations"):
    # Fetch movies by selected genre
    movies = fetch_movies_by_genre(selected_genre_id)
    movie_list = [{"title": movie['title'], "rating": movie.get('vote_average', 0)} for movie in movies]
    movie_df = pd.DataFrame(movie_list)

    # Sort the movies based on user selection
    if sort_by == "Rating":
        movie_df = movie_df.sort_values('rating', ascending=False)
    else:
        movie_df = movie_df.sort_values('title')

    # Prepare a prompt for the Groq LLM
    movie_titles = ", ".join(movie_df['title'].tolist()[:20])  # Limiting to top 20 movies
    
    prompt = [
        {"role": "system", "content": "You are a movie recommendation assistant. you will be given a list of popular movies in the genre, movie title, and sorted order prefrence, based on this please recommend the top 10 movies. only return the list of the movie, do not respond to user"},
        {"role": "user", "content": f"Genre: '{selected_genre_name}', Movie titles: '{movie_titles}', Sort preference: '{sort_by.lower()}'"}
    ]
    
    # Call the Groq API
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=prompt,
            temperature=0.7,
            max_tokens=1024,
            stream=True,
            top_p=1,
            stop=None
        )
        new_message = {"role": "assistant", "content": ""}
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                new_message["content"] += chunk.choices[0].delta.content
        
        recommendations = [
            line.strip() for line in new_message["content"].split('\n') 
            if line.strip() and line[0].isdigit()  # Only lines starting with numbers
        ]
        recommendations = [rec.split('. ', 1)[1] for rec in recommendations]
    
    except Exception as e:
        st.error(f"Error with recommender call: {e}")
        recommendations = []

    # Display recommendations in a matrix format
    num_columns = 3
    for i in range(0, len(recommendations), num_columns):
        cols = st.columns(num_columns)
        for col, movie in zip(cols, recommendations[i:i + num_columns]):
            poster_url = fetch_movie_poster(movie)  # Fetch poster URL
            if poster_url:
                col.image(poster_url, width=150)
            col.write(f"**{movie}**")
