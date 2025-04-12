import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from difflib import get_close_matches
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# TMDB API configuration
TMDB_API_KEY = '3c6ec703890682871f09009db1489504'
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# Load and prepare data
def load_data():
    try:
        # Read the CSV files
        logger.info("Loading CSV files...")
        df1 = pd.read_csv('cleaned_movies_credits.csv')
        df2 = pd.read_csv('cleaned_movies_metadata.csv')
        
        # Print column names and first few rows for debugging
        logger.info(f"Credits DataFrame columns: {df1.columns.tolist()}")
        logger.info(f"Metadata DataFrame columns: {df2.columns.tolist()}")
        
        # Rename columns in credits DataFrame to match expected names
        if 'movie_id' in df1.columns:
            df1 = df1.rename(columns={'movie_id': 'id'})
            logger.info("Renamed 'movie_id' to 'id' in credits DataFrame")
        
        # Merge the dataframes
        logger.info("Merging dataframes...")
        df = df2.merge(df1, on='id', how='inner')
        logger.info(f"Merged DataFrame shape: {df.shape}")
        logger.info(f"Merged DataFrame columns: {df.columns.tolist()}")
        
        # Fix duplicated title columns
        if 'title_x' in df.columns and 'title_y' in df.columns:
            # Use title_x as the main title column
            df = df.rename(columns={'title_x': 'title'})
            # Drop title_y as it's redundant
            df = df.drop(columns=['title_y'])
            logger.info("Fixed duplicated title columns")
        
        # Calculate weighted ratings
        logger.info("Calculating weighted ratings...")
        C = df['vote_average'].mean()
        m = df['vote_count'].quantile(0.9)
        logger.info(f"Average vote: {C}, Vote count threshold: {m}")
        
        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            return (v/(v+m) * R) + (m/(m+v) * C)
        
        df['score'] = df.apply(weighted_rating, axis=1)
        
        # Prepare TF-IDF matrix for content-based recommendations
        logger.info("Preparing TF-IDF matrix...")
        df['overview'] = df['overview'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['overview'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        logger.info("Data loading completed successfully!")
        return df, cosine_sim
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Load data at startup
try:
    df, cosine_sim = load_data()
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    logger.info(f"Loaded {len(indices)} unique movie titles")
except Exception as e:
    logger.error(f"Failed to initialize data: {str(e)}")
    df = None
    cosine_sim = None
    indices = None

def get_tmdb_movie_details(movie_id):
    """Fetch movie details from TMDB API with retry logic"""
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'append_to_response': 'credits,images'
        }
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching TMDB details for movie {movie_id}: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error fetching TMDB details for movie {movie_id}: {str(e)}")
        return {}

def get_movie_poster(movie_id):
    """Get movie poster URL from TMDB with retry logic"""
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}/images"
        params = {'api_key': TMDB_API_KEY}
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('posters'):
            return TMDB_IMAGE_BASE_URL + data['posters'][0]['file_path']
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching poster for movie {movie_id}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching poster for movie {movie_id}: {str(e)}")
        return None

def get_top_movies(top_n=10):
    """Get top movies by weighted rating"""
    if df is None:
        logger.error("DataFrame is None, cannot get top movies")
        return []
    
    try:
        logger.info(f"Getting top {top_n} movies by score")
        top_movies = df.sort_values('score', ascending=False).head(top_n)
        logger.info(f"Found {len(top_movies)} top movies")
        
        movies_data = []
        
        for _, movie in top_movies.iterrows():
            movie_data = {
                'title': movie['title'],
                'overview': movie['overview'],
                'vote_average': movie['vote_average'],
                'score': movie['score'],
                'id': movie['id']
            }
            
            # Get TMDB details
            tmdb_details = get_tmdb_movie_details(movie['id'])
            movie_data['poster_path'] = get_movie_poster(movie['id'])
            movie_data['release_date'] = tmdb_details.get('release_date', '')
            movie_data['runtime'] = tmdb_details.get('runtime', 0)
            
            movies_data.append(movie_data)
        
        logger.info(f"Returning {len(movies_data)} movies with details")
        return movies_data
    except Exception as e:
        logger.error(f"Error getting top movies: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def find_movie_by_title(title):
    """Find a movie by title using fuzzy matching if exact match fails"""
    if df is None:
        return None
    
    # Try exact match first
    if title in indices:
        return title
    
    # Try case-insensitive match
    title_lower = title.lower()
    for movie_title in indices.index:
        if movie_title.lower() == title_lower:
            return movie_title
    
    # Try fuzzy matching
    all_titles = indices.index.tolist()
    matches = get_close_matches(title, all_titles, n=1, cutoff=0.8)
    if matches:
        logger.info(f"Fuzzy matched '{title}' to '{matches[0]}'")
        return matches[0]
    
    return None

def content_based_recommendations(title, cosine_sim=cosine_sim, top_n=10):
    """Content-based recommendations using movie overview"""
    if df is None or cosine_sim is None:
        logger.error("DataFrame or cosine similarity matrix is None")
        return []
    
    # Find the movie using fuzzy matching
    matched_title = find_movie_by_title(title)
    if not matched_title:
        logger.warning(f"No match found for title: {title}")
        return []
    
    try:
        idx = indices[matched_title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]
        
        recommendations = df.iloc[movie_indices][['title', 'id', 'overview', 'vote_average']].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        logger.info(f"Found {len(recommendations)} content-based recommendations for '{matched_title}'")
        return recommendations.to_dict('records')
    except Exception as e:
        logger.error(f"Error in content-based recommendations: {str(e)}")
        return []

def genre_based_recommendations(title, top_n=10):
    """Genre-based recommendations"""
    if df is None:
        logger.error("DataFrame is None")
        return []
    
    # Find the movie using fuzzy matching
    matched_title = find_movie_by_title(title)
    if not matched_title:
        logger.warning(f"No match found for title: {title}")
        return []
    
    try:
        movie_genres = df[df['title'] == matched_title]['genres'].iloc[0]
        if isinstance(movie_genres, str):
            movie_genres = eval(movie_genres)
        
        # Find movies with similar genres
        similar_movies = df[df['genres'].apply(lambda x: any(g in eval(x) for g in movie_genres))]
        similar_movies = similar_movies[similar_movies['title'] != matched_title]
        similar_movies = similar_movies.sort_values('score', ascending=False)
        
        recommendations = similar_movies.head(top_n)[['title', 'id', 'overview', 'vote_average']].to_dict('records')
        logger.info(f"Found {len(recommendations)} genre-based recommendations for '{matched_title}'")
        return recommendations
    except Exception as e:
        logger.error(f"Error in genre-based recommendations: {str(e)}")
        return []

@app.route('/')
def home():
    top_movies = get_top_movies()
    logger.info(f"Rendering home page with {len(top_movies)} top movies")
    return render_template('index.html', top_movies=top_movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    if df is None:
        logger.error("Movie database is not initialized properly")
        return jsonify({'error': 'Movie database is not initialized properly'})
    
    movie_title = request.form.get('movie_title')
    if not movie_title:
        logger.warning("No movie title provided")
        return jsonify({'error': 'Please provide a movie title'})
    
    logger.info(f"Received recommendation request for: {movie_title}")
    
    try:
        # Get recommendations from different methods
        content_recs = content_based_recommendations(movie_title)
        genre_recs = genre_based_recommendations(movie_title)
        
        # Process content-based recommendations
        processed_content_recs = []
        for rec in content_recs:
            tmdb_details = get_tmdb_movie_details(rec['id'])
            rec['poster_path'] = get_movie_poster(rec['id'])
            rec['release_date'] = tmdb_details.get('release_date', '')
            rec['runtime'] = tmdb_details.get('runtime', 0)
            processed_content_recs.append(rec)
        
        # Process genre-based recommendations
        processed_genre_recs = []
        for rec in genre_recs:
            tmdb_details = get_tmdb_movie_details(rec['id'])
            rec['poster_path'] = get_movie_poster(rec['id'])
            rec['release_date'] = tmdb_details.get('release_date', '')
            rec['runtime'] = tmdb_details.get('runtime', 0)
            processed_genre_recs.append(rec)
        
        logger.info(f"Returning {len(processed_content_recs)} content-based and {len(processed_genre_recs)} genre-based recommendations")
        return jsonify({
            'content_based': processed_content_recs[:5],  # Top 5 content-based recommendations
            'genre_based': processed_genre_recs[:5]       # Top 5 genre-based recommendations
        })
    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An error occurred while processing your request'})

if __name__ == '__main__':
    app.run(debug=True) 