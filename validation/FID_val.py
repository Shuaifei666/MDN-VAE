import numpy as np
import networkx as nx
from scipy.linalg import sqrtm

def compute_graph_features(matrix):
    # 矩阵归一化
    max_value = matrix.max()
    if max_value > 0:
        matrix_normalized = matrix / max_value
    else:
        matrix_normalized = np.zeros_like(matrix)

    # 将连接强度转换为距离
    epsilon = 1e-5
    distance_matrix = 1 / (matrix_normalized + epsilon)

    # 创建加权图
    G = nx.from_numpy_array(matrix_normalized)

    # 计算加权度数
    degrees = np.array([val for (node, val) in G.degree(weight='weight')])
    degree_mean = degrees.mean()
    degree_std = degrees.std()

    # 计算加权聚类系数
    clustering_coeffs = np.array(list(nx.clustering(G, weight='weight').values()))
    clustering_mean = clustering_coeffs.mean()
    clustering_std = clustering_coeffs.std()

    # 添加边的距离属性
    edge_distances = {}
    for i, j in G.edges():
        edge_distances[(i, j)] = distance_matrix[i, j]
    nx.set_edge_attributes(G, edge_distances, 'distance')

    # 计算加权平均最短路径长度
    try:
        avg_shortest_path_length = nx.average_shortest_path_length(G, weight='distance')
    except nx.NetworkXError:
        # 处理不连通图的情况
        lengths = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph) > 1:
                length = nx.average_shortest_path_length(subgraph, weight='distance')
                lengths.append(length)
        avg_shortest_path_length = np.mean(lengths) if lengths else 0

    # 特征向量
    features = np.array([
        degree_mean,
        degree_std,
        clustering_mean,
        clustering_std,
        avg_shortest_path_length
    ])
    return features

def calculate_fid(features_real, features_generated):
    # 计算均值和协方差矩阵
    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)

    mu_gen = np.mean(features_generated, axis=0)
    sigma_gen = np.cov(features_generated, rowvar=False)

    # 计算均值差的平方
    diff = mu_real - mu_gen
    mean_diff = diff.dot(diff)

    # 计算协方差矩阵的平方根
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    # 处理可能的复数结果
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算 FID
    fid = mean_diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid
import numpy as np
import networkx as nx
from scipy.linalg import sqrtm

# 您之前的函数 compute_graph_features 和 calculate_fid 保持不变

def generate_sbm_graphs(num_graphs, N, K, pi, P, seed=None):
    graphs = []
    np.random.seed(seed)
    P = np.array(P)  # 将 P 转换为 NumPy 数组
    for _ in range(num_graphs):
        # 生成 SBM 图
        blocks = np.random.choice(K, size=N, p=pi)
        adj_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                prob = P[blocks[i], blocks[j]]  # 现在可以正常索引
                if np.random.rand() < prob:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        graphs.append(adj_matrix)
    return graphs


# 参数设置
num_graphs = 50
N = 100
K = 3
pi = [0.3, 0.4, 0.3]
P_real = [
    [0.8, 0.05, 0.02],
    [0.05, 0.7, 0.03],
    [0.02, 0.03, 0.6]
]

# 生成真实图
real_matrices = generate_sbm_graphs(num_graphs, N, K, pi, P_real, seed=42)

# 生成少量噪声的图
P_noisy_small = [
    [0.8, 0.10, 0.07],
    [0.10, 0.7, 0.08],
    [0.07, 0.08, 0.6]
]
generated_matrices_small_noise = generate_sbm_graphs(num_graphs, N, K, pi, P_noisy_small, seed=43)

# 生成大量噪声的图
P_noisy_large = [
    [0.8, 0.3, 0.3],
    [0.3, 0.7, 0.3],
    [0.3, 0.3, 0.6]
]
generated_matrices_large_noise = generate_sbm_graphs(num_graphs, N, K, pi, P_noisy_large, seed=44)

# 提取图论特征
features_real = [compute_graph_features(matrix) for matrix in real_matrices]
features_generated_small_noise = [compute_graph_features(matrix) for matrix in generated_matrices_small_noise]
features_generated_large_noise = [compute_graph_features(matrix) for matrix in generated_matrices_large_noise]

# 计算 FID
fid_small_noise = calculate_fid(features_real, features_generated_small_noise)
fid_large_noise = calculate_fid(features_real, features_generated_large_noise)

print("真实图 vs. 少量噪声的生成图的 FID 值：", fid_small_noise)
print("真实图 vs. 大量噪声的生成图的 FID 值：", fid_large_noise)
