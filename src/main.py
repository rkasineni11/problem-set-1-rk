import os
import analysis_network_centrality
import analysis_similar_actors_genre

# Define the file paths
output_directory = '/Users/KasineniFamily/problem-set-1-rk/problem-set-1-rk/'
file_path = os.path.join(output_directory, "imdb.json")

# Call functions / instantiate objects from the two analysis .py files
def main():
    # Perform network centrality analysis
    print("Starting Network Centrality Analysis...")
    
    # Load and process the IMDb dataset for network centrality analysis
    g = analysis_network_centrality.load_and_process_data(file_path)
    
    # Calculate centrality and output the results
    analysis_network_centrality.output_centrality_and_edges(g, output_directory)
    
    print("Network Centrality Analysis Completed.")
    
    # Perform similar actors by genre analysis
    print("Starting Similar Actors by Genre Analysis...")
    
    # Load the data and construct the feature matrix
    df, actor_name_mapping = analysis_similar_actors_genre.load_data(file_path)
    
    # Example: Find similar actors to Chris Hemsworth (actor ID 'nm1165110')
    query_actor_id = 'nm1165110'
    
    # Find and save similar actors based on Cosine distance
    similar_actors_cosine = analysis_similar_actors_genre.find_similar_actors(df, query_actor_id, actor_name_mapping, metric='cosine')
    analysis_similar_actors_genre.save_similar_actors(similar_actors_cosine, actor_name_mapping[query_actor_id], output_directory, metric='cosine')

    # Find and save similar actors based on Euclidean distance
    similar_actors_euclidean = analysis_similar_actors_genre.find_similar_actors(df, query_actor_id, actor_name_mapping, metric='euclidean')
    analysis_similar_actors_genre.save_similar_actors(similar_actors_euclidean, actor_name_mapping[query_actor_id], output_directory, metric='euclidean')
    
    print("Similar Actors by Genre Analysis Completed.")

if __name__ == "__main__":
    main()
