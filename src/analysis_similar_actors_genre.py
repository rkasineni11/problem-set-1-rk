import json
import pandas as pd
from sklearn.metrics import pairwise_distances
from datetime import datetime

def load_data(file_path):
    """
    Load the JSON data and construct a feature matrix where each row corresponds to an actor and each column to a genre.
    
    Args:
    file_path (str): Path to the JSON file.

    Returns:
    pd.DataFrame: Feature matrix with actors as rows and genres as columns.
    dict: Mapping from actor IDs to actor names.
    """
    genre_set = set()
    actor_genre_count = {}
    actor_name_mapping = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                movie_data = json.loads(line)
                genres = movie_data.get('genres', [])
                actors = movie_data.get('actors', [])

                for actor_id, actor_name in actors:
                    if actor_id not in actor_genre_count:
                        actor_genre_count[actor_id] = {}
                        actor_name_mapping[actor_id] = actor_name

                    for genre in genres:
                        genre_set.add(genre)
                        if genre not in actor_genre_count[actor_id]:
                            actor_genre_count[actor_id][genre] = 0
                        actor_genre_count[actor_id][genre] += 1
            except json.JSONDecodeError:
                continue  # Skip lines that are not valid JSON

    # Create DataFrame
    genres = list(genre_set)
    df = pd.DataFrame(index=actor_genre_count.keys(), columns=genres).fillna(0)

    for actor_id, genre_counts in actor_genre_count.items():
        for genre, count in genre_counts.items():
            df.at[actor_id, genre] = count

    return df, actor_name_mapping

def find_similar_actors(df, actor_id, actor_name_mapping, metric='cosine', top_n=10):
    """
    Find the top N most similar actors to the given actor based on genre appearances.
    
    Args:
    df (pd.DataFrame): Feature matrix with actors as rows and genres as columns.
    actor_id (str): Actor ID of the query actor.
    actor_name_mapping (dict): Mapping from actor IDs to actor names.
    metric (str): Distance metric to use ('cosine' or 'euclidean').
    top_n (int): Number of similar actors to find.

    Returns:
    pd.DataFrame: DataFrame containing the top N most similar actors.
    """
    distances = pairwise_distances(df, df.loc[[actor_id]], metric=metric).flatten()
    distance_series = pd.Series(distances, index=df.index)
    similar_actors = distance_series.nsmallest(top_n + 1).iloc[1:]  # Exclude the actor itself

    similar_actors_df = pd.DataFrame({
        'Actor_ID': similar_actors.index,
        'Actor_Name': [actor_name_mapping[aid] for aid in similar_actors.index],
        'Distance': similar_actors.values
    })

    return similar_actors_df

def save_similar_actors(similar_actors_df, actor_name, output_directory, metric):
    """
    Save the similar actors DataFrame to a CSV file.
    
    Args:
    similar_actors_df (pd.DataFrame): DataFrame containing similar actors.
    actor_name (str): Name of the query actor.
    output_directory (str): Directory to save the output CSV file.
    metric (str): Distance metric used.
    """
    file_name = f'similar_actors_{actor_name}_{metric}_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    output_path = f'{output_directory}/{file_name}'
    similar_actors_df.to_csv(output_path, index=False)
    print(f"Similar actors based on {metric} distance saved to: {output_path}")
