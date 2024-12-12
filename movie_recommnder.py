import pickle as pk
import pandas as pd
import numpy as np

class MovieRecommenderTest:
    def __init__(self):
        self.movies_data = None
        self.cosine_sim = None
    
    def load_data(self, file_prefix='movie_data'):
        """Load DataFrame and similarity matrix from pickle files"""
        try:
            print("Loading movie data from pickle files...")
            with open(f'{file_prefix}_df.pkl', 'rb') as f:
                self.movies_data = pk.load(f)
            with open(f'{file_prefix}_sim.pkl', 'rb') as f:
                self.cosine_sim = pk.load(f)
            
            # Print verification info
            print(f"\nLoaded {len(self.movies_data)} movies successfully!")
            print("\nSample of available movies:")
            print(self.movies_data['title'].head())
            
            # Verify data types
            print("DataFrame columns:", self.movies_data.columns.tolist())
            print("Similarity matrix shape:", self.cosine_sim.shape)
            
            return True
            
        except FileNotFoundError:
            print("\nError: Pickle files not found!")
            print("Make sure 'movie_data_df.pkl' and 'movie_data_sim.pkl' are in the same directory.")
            return False
        except Exception as e:
            print(f"\nError loading data: e")
            return False
    
    def get_recommendations(self, movie_title, n=5):
        """Get movie recommendations based on similarity"""
        if self.movies_data is None or self.cosine_sim is None:
            return "Error: Data not loaded. Please load data first."
        
        # Make the search case-insensitive and partial match
        mask = self.movies_data['title'].str.lower().str.contains(movie_title.lower())
        matches = self.movies_data[mask]
        
        if matches.empty:
            return f"No movies found matching '{movie_title}'"
        
        # Use the first matching movie
        idx = matches.index[0]
        movie_found = matches.iloc[0]['title']
        print(f"\nFound movie: {movie_found}")
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar movies (excluding itself)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        
        recommendations = self.movies_data.iloc[movie_indices][['title', 'genres', 'release_date']]
        return recommendations

def main():
    # Create recommender instance
    recommender = MovieRecommenderTest()
    
    # Load the pre-existing data
    if not recommender.load_data():
        print("\nExiting due to data loading error.")
        return
    
    # Interactive recommendation loop
    while True:
        print("\n" + "="*50)
        print("Movie Recommendation System")
        print("="*50)
        print("\nEnter 'quit' to exit")
        
        # Get movie title from user
        movie_title = input("\nEnter movie title (or partial name): ").strip()
        
        if movie_title.lower() == 'quit':
            print("\nThank you for using the Movie Recommender!")
            break
        
        # Get and display recommendations
        print("\nFinding recommendations...")
        recommendations = recommender.get_recommendations(movie_title)
        print("\nRecommended movies:")
        print(recommendations)

if __name__ == "__main__":
    main()
