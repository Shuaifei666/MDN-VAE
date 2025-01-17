import numpy as np
import networkx as nx
from scipy.linalg import sqrtm

# Function to compute graph features
def compute_graph_features(matrix):
    # Normalize the matrix
    max_value = matrix.max()
    if max_value > 0:
        matrix_normalized = matrix / max_value
    else:
        matrix_normalized = np.zeros_like(matrix)

    # Convert connection strength to distance
    epsilon = 1e-5
    distance_matrix = 1 / (matrix_normalized + epsilon)

    # Create a weighted graph from the matrix
    G = nx.from_numpy_array(matrix_normalized)

    # Compute weighted degrees
    degrees = np.array([val for (node, val) in G.degree(weight='weight')])
    degree_mean = degrees.mean()
    degree_std = degrees.std()

    # Compute weighted clustering coefficients
    clustering_coeffs = np.array(list(nx.clustering(G, weight='weight').values()))
    clustering_mean = clustering_coeffs.mean()
    clustering_std = clustering_coeffs.std()

    # Add distance attributes to edges
    edge_distances = {}
    for i, j in G.edges():
        edge_distances[(i, j)] = distance_matrix[i, j]
    nx.set_edge_attributes(G, edge_distances, 'distance')

    # Compute weighted average shortest path length
    try:
        avg_shortest_path_length = nx.average_shortest_path_length(G, weight='distance')
    except nx.NetworkXError:
        # Handle disconnected graphs
        lengths = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph) > 1:
                length = nx.average_shortest_path_length(subgraph, weight='distance')
                lengths.append(length)
        avg_shortest_path_length = np.mean(lengths) if lengths else 0

    # Feature vector
    features = np.array([
        degree_mean,
        degree_std,
        clustering_mean,
        clustering_std,
        avg_shortest_path_length
    ])
    return features

# Function to calculate FID (Fréchet Inception Distance)
def calculate_fid(features_real, features_generated):
    # Compute mean and covariance matrix
    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)

    mu_gen = np.mean(features_generated, axis=0)
    sigma_gen = np.cov(features_generated, rowvar=False)

    # Compute the squared difference of means
    diff = mu_real - mu_gen
    mean_diff = diff.dot(diff)

    # Compute the square root of the covariance matrix product
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    # Handle potential complex values
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute FID
    fid = mean_diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

# Function to generate stochastic block model (SBM) graphs
def generate_sbm_graphs(num_graphs, N, K, pi, P, seed=None):
    graphs = []
    np.random.seed(seed)
    P = np.array(P)  # Convert P to a NumPy array
    for _ in range(num_graphs):
        # Generate SBM graph
        blocks = np.random.choice(K, size=N, p=pi)
        adj_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                prob = P[blocks[i], blocks[j]]  # Access probability matrix
                if np.random.rand() < prob:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        graphs.append(adj_matrix)
    return graphs

# Parameters
num_graphs = 50
N = 100
K = 3
pi = [0.3, 0.4, 0.3]
P_real = [
    [0.8, 0.05, 0.02],
    [0.05, 0.7, 0.03],
    [0.02, 0.03, 0.6]
]

# Generate real graphs
real_matrices = generate_sbm_graphs(num_graphs, N, K, pi, P_real, seed=42)

# Generate graphs with small noise
P_noisy_small = [
    [0.8, 0.10, 0.07],
    [0.10, 0.7, 0.08],
    [0.07, 0.08, 0.6]
]
generated_matrices_small_noise = generate_sbm_graphs(num_graphs, N, K, pi, P_noisy_small, seed=43)

# Generate graphs with large noise
P_noisy_large = [
    [0.8, 0.3, 0.3],
    [0.3, 0.7, 0.3],
    [0.3, 0.3, 0.6]
]
generated_matrices_large_noise = generate_sbm_graphs(num_graphs, N, K, pi, P_noisy_large, seed=44)

# Extract graph features
features_real = [compute_graph_features(matrix) for matrix in real_matrices]
features_generated_small_noise = [compute_graph_features(matrix) for matrix in generated_matrices_small_noise]
features_generated_large_noise = [compute_graph_features(matrix) for matrix in generated_matrices_large_noise]

# Calculate FID
fid_small_noise = calculate_fid(features_real, features_generated_small_noise)
fid_large_noise = calculate_fid(features_real, features_generated_large_noise)

# Print FID values
print("FID between real graphs and graphs with small noise:", fid_small_noise)
print("FID between real graphs and graphs with large noise:", fid_large_noise)
