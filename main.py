import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests

# Import API key from config file
try:
    from config import TMDB_API_KEY, TMDB_API_BASE_URL, TMDB_IMAGE_BASE_URL, TMDB_PROFILE_IMAGE_BASE_URL
except ImportError:
    print("Warning: config.py not found. Please create config.py with your TMDB_API_KEY")
    TMDB_API_KEY = ""
    TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
    TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
    TMDB_PROFILE_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w185"

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! try another movie name')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

# Function to get movie poster and details from TMDB
def get_movie_poster(movie_title):
    """
    Fetch movie poster and details from TMDB API
    Returns: dict with poster_path, year, rating, and overview
    """
    if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key_here":
        print("ERROR: TMDB API key not configured")
        return None
    
    try:
        # Search for the movie
        search_url = f"{TMDB_API_BASE_URL}/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': movie_title
        }
        
        response = requests.get(search_url, params=params, timeout=5)
        data = response.json()
        
        if data.get('results') and len(data['results']) > 0:
            movie_data = data['results'][0]  # Get the first result
            
            return {
                'title': movie_data.get('title', movie_title),
                'poster_path': f"{TMDB_IMAGE_BASE_URL}{movie_data['poster_path']}" if movie_data.get('poster_path') else None,
                'year': movie_data.get('release_date', '')[:4] if movie_data.get('release_date') else 'N/A',
                'overview': movie_data.get('overview', ''),
                'rating': round(movie_data.get('vote_average', 0), 1)
            }
    except Exception as e:
        print(f"Error fetching movie details for '{movie_title}': {e}")
    
    return None

# NEW: Function to search for actors
def search_actor(actor_name):
    """
    Search for an actor and return their details
    """
    if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key_here":
        print("ERROR: TMDB API key not configured")
        return None
    
    try:
        search_url = f"{TMDB_API_BASE_URL}/search/person"
        params = {
            'api_key': TMDB_API_KEY,
            'query': actor_name
        }
        
        print(f"Searching for actor: {actor_name}")
        response = requests.get(search_url, params=params, timeout=10)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"API Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        data = response.json()
        print(f"API Response: {data}")
        
        if data.get('results') and len(data['results']) > 0:
            actors = []
            for actor in data['results'][:5]:  # Return top 5 matches
                actors.append({
                    'id': actor.get('id'),
                    'name': actor.get('name'),
                    'profile_path': f"{TMDB_PROFILE_IMAGE_BASE_URL}{actor['profile_path']}" if actor.get('profile_path') else None,
                    'known_for_department': actor.get('known_for_department', 'Acting'),
                    'popularity': actor.get('popularity', 0)
                })
            print(f"Found {len(actors)} actors")
            return actors
        else:
            print("No results found")
            return None
    except Exception as e:
        print(f"Error searching for actor '{actor_name}': {e}")
        import traceback
        traceback.print_exc()
    
    return None

# NEW: Function to get actor's movies
def get_actor_movies(actor_id, limit=50):
    """
    Get top movies for a specific actor
    """
    if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key_here":
        print("ERROR: TMDB API key not configured")
        return None
    
    try:
        # Get actor's movie credits
        credits_url = f"{TMDB_API_BASE_URL}/person/{actor_id}/movie_credits"
        params = {
            'api_key': TMDB_API_KEY
        }
        
        print(f"Fetching movies for actor ID: {actor_id}")
        response = requests.get(credits_url, params=params, timeout=10)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"API Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        data = response.json()
        
        if data.get('cast'):
            # Sort by popularity and vote_average
            movies = data['cast']
            print(f"Found {len(movies)} movies for actor")
            
            # Filter out movies without release dates and sort by popularity
            movies = [m for m in movies if m.get('release_date')]
            movies = sorted(movies, key=lambda x: (x.get('popularity', 0), x.get('vote_average', 0)), reverse=True)
            
            # Get top N movies
            top_movies = []
            for movie in movies[:limit]:
                top_movies.append({
                    'id': movie.get('id'),
                    'title': movie.get('title'),
                    'character': movie.get('character', 'Unknown'),
                    'poster_path': f"{TMDB_IMAGE_BASE_URL}{movie['poster_path']}" if movie.get('poster_path') else None,
                    'year': movie.get('release_date', '')[:4] if movie.get('release_date') else 'N/A',
                    'rating': round(movie.get('vote_average', 0), 1),
                    'overview': movie.get('overview', ''),
                    'popularity': movie.get('popularity', 0)
                })
            
            print(f"Returning {len(top_movies)} top movies")
            return top_movies
        else:
            print("No cast data found")
            return None
    except Exception as e:
        print(f"Error fetching movies for actor ID {actor_id}: {e}")
        import traceback
        traceback.print_exc()
    
    return None

# NEW: Function to get actor details
def get_actor_details(actor_id):
    """
    Get detailed information about an actor
    """
    if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key_here":
        print("ERROR: TMDB API key not configured")
        return None
    
    try:
        details_url = f"{TMDB_API_BASE_URL}/person/{actor_id}"
        params = {
            'api_key': TMDB_API_KEY
        }
        
        print(f"Fetching details for actor ID: {actor_id}")
        response = requests.get(details_url, params=params, timeout=10)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"API Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        data = response.json()
        
        actor_details = {
            'id': data.get('id'),
            'name': data.get('name'),
            'biography': data.get('biography', 'No biography available'),
            'birthday': data.get('birthday', 'Unknown'),
            'place_of_birth': data.get('place_of_birth', 'Unknown'),
            'profile_path': f"{TMDB_IMAGE_BASE_URL}{data['profile_path']}" if data.get('profile_path') else None,
            'known_for_department': data.get('known_for_department', 'Acting'),
            'popularity': data.get('popularity', 0)
        }
        print(f"Actor details fetched successfully: {actor_details['name']}")
        return actor_details
    except Exception as e:
        print(f"Error fetching actor details for ID {actor_id}: {e}")
        import traceback
        traceback.print_exc()
    
    return None
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

# to get suggestions of movies
def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

# Flask API
app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return jsonify({'error': rc})
    else:
        # Get movie details with posters for each recommendation
        movie_details_list = []
        for movie_title in rc:
            details = get_movie_poster(movie_title)
            if details:
                movie_details_list.append(details)
            else:
                # Fallback if TMDB doesn't have the movie or API key not set
                movie_details_list.append({
                    'title': movie_title,
                    'poster_path': None,
                    'year': 'N/A',
                    'overview': '',
                    'rating': 0
                })
        
        return jsonify(movie_details_list)

# NEW: Actor search endpoint
@app.route("/search_actor", methods=["POST"])
def search_actor_endpoint():
    try:
        actor_name = request.form.get('name')
        print(f"Received search request for: {actor_name}")
        
        if not actor_name:
            return jsonify({'error': 'Please provide an actor name'}), 400
        
        # Check API key
        if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key_here":
            return jsonify({'error': 'TMDB API key not configured. Please add your API key to config.py'}), 500
        
        actors = search_actor(actor_name)
        if actors:
            return jsonify(actors), 200
        else:
            return jsonify({'error': 'Actor not found. Please try another name.'}), 404
            
    except Exception as e:
        print(f"Error in search_actor_endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# NEW: Get actor's movies endpoint
@app.route("/actor_movies/<int:actor_id>", methods=["GET"])
def actor_movies_endpoint(actor_id):
    try:
        limit = request.args.get('limit', 50, type=int)
        print(f"Received request for actor ID {actor_id} movies")
        
        # Check API key
        if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key_here":
            return jsonify({'error': 'TMDB API key not configured. Please add your API key to config.py'}), 500
        
        # Get actor details
        actor_details = get_actor_details(actor_id)
        
        # Get actor's movies
        movies = get_actor_movies(actor_id, limit)
        
        if movies and actor_details:
            return jsonify({
                'actor': actor_details,
                'movies': movies
            }), 200
        else:
            return jsonify({'error': 'Could not fetch actor movies'}), 404
            
    except Exception as e:
        print(f"Error in actor_movies_endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}

    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)

if __name__ == '__main__':
    app.run(debug=True)