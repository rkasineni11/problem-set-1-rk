import json
import networkx as nx
import pandas as pd
from datetime import datetime

# Function to load and process the IMDb dataset
def load_and_process_data(file_path):
    """
    Loads and processes the IMDb dataset, building a graph with actors as nodes and their collaborations as edges.

    Args:
    file_path (str): Path to the JSON file containing the IMDb dataset.

    Returns:
    nx.Graph: The constructed graph with actors and their collaborations.
    """
    g = nx.Graph()

    # Read and process the IMDb dataset
    with open(file_path, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            try:
                this_movie = json.loads(line)
                process_movie_data(this_movie, g)
            except json.JSONDecodeError:
                continue  # Skip lines that are not valid JSON
    
    return g

def process_movie_data(movie_data, graph):
    """
    Processes a single movie's data, adding nodes and edges to the graph.
    
    Args:
    movie_data (dict): Dictionary containing the movie's actor information.
    graph (nx.Graph): Graph to which nodes and edges will be added.
    """
    actors = movie_data.get('actors', [])
    
    # Create a node for every actor and add edges between every pair of actors
    for i, (left_actor_id, left_actor_name) in enumerate(actors):
        if left_actor_name not in graph:
            graph.add_node(left_actor_name)
        
        for j in range(i + 1, len(actors)):
            right_actor_id, right_actor_name = actors[j]
            if right_actor_name not in graph:
                graph.add_node(right_actor_name)
            
            # Add or update the edge between the two actors
            if graph.has_edge(left_actor_name, right_actor_name):
                graph[left_actor_name][right_actor_name]['weight'] += 1
            else:
                graph.add_edge(left_actor_name, right_actor_name, weight=1)

def output_centrality_and_edges(graph, output_directory):
    """
    Calculates degree centrality of the graph, outputs both centrality metrics and edge list to CSV files.

    Args:
    graph (nx.Graph): The graph containing actors and their collaborations.
    output_directory (str): Directory to save the output CSV files.
    """
    # Print the number of nodes
    print("Nodes:", len(graph.nodes))

    # Calculate and print the 10 most central nodes based on degree centrality
    centrality = nx.degree_centrality(graph)
    top_10_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 Most Central Nodes:")
    for actor, cent in top_10_central:
        print(f"{actor}: {cent}")

    # Prepare the final dataframe for output
    edges_data = [(u, "<->", v, d['weight']) for u, v, d in graph.edges(data=True)]
    df_edges = pd.DataFrame(edges_data, columns=['left_actor_name', '<->', 'right_actor_name', 'weight'])

    # Output the final dataframe to a CSV
    output_file_name = f'network_centrality_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    df_edges.to_csv(f'{output_directory}/{output_file_name}', index=False)

    print(f"Edge list saved to: {output_directory}/{output_file_name}")
