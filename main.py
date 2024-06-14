import pandas as pd

# Load the data
data = pd.read_csv('ratings_small.csv')

# Display the first few rows of the data
data.head()

# Load the movie metadata
movies_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Display the first few rows of the movie metadata
movies_metadata.head()

# Handle missing values
movies_metadata['overview'] = movies_metadata['overview'].fillna('')
movies_metadata['tagline'] = movies_metadata['tagline'].fillna('')
movies_metadata['genres'] = movies_metadata['genres'].fillna('')

# Extract genres from the genres column
def extract_genres(genres_str):
    try:
        genres = eval(genres_str)
        return ' '.join([g['name'] for g in genres])
    except:
        return ''

movies_metadata['genres_str'] = movies_metadata['genres'].apply(extract_genres)

# Combine genres, overview, and tagline into a single string
movies_metadata['combined_text'] = movies_metadata['genres_str'] + ' ' + movies_metadata['overview'] + ' ' + movies_metadata['tagline']

# Display the first few rows of the combined text
movies_metadata[['title', 'combined_text']].head()

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Compute the TF-IDF matrix for the combined text
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_metadata['combined_text'])

# Display the shape of the TF-IDF matrix
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
#cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_movie_recommendations(title, cosine_sim_matrix=None, top_n=10):
    # Get the index of the movie from its title
    idx = movies_metadata[movies_metadata['title'] == title].index[0]
    
    # If a precomputed cosine similarity matrix is provided, use it
    if cosine_sim_matrix is not None:
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    else:
        # Otherwise, compute the cosine similarity scores on-the-fly
        cosine_sim_vector = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        sim_scores = list(enumerate(cosine_sim_vector))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N most similar movies' indices
    sim_movie_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    # Return the top N most similar movies
    return movies_metadata['title'].iloc[sim_movie_indices]

# Test the recommendation function with a sample movie
sample_movie = "Star Wars"
recommendations = get_movie_recommendations(sample_movie)
recommendations
