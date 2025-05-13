import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import gzip
import random
import os  
from networkx.algorithms import community
from typing import Dict, List, Tuple, Set
from matplotlib.font_manager import FontProperties

# 添加字体路径 (确保路径正确)
font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
try:
    font_prop = FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
except IOError:
    print(f"警告: 字体文件 {font_path} 未找到，可能无法正常显示中文。")

# -------------------- 核心功能函数 --------------------

def load_graph_from_edgelist(filepath: str) -> nx.Graph:
    """从边列表文件加载图数据."""
    print(f"加载图数据: {filepath}")
    with gzip.open(filepath, 'rt') as f:
        G = nx.parse_edgelist(f, delimiter='\t', create_using=nx.Graph(), nodetype=int)
    print(f"已加载图，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
    return G

def get_largest_connected_component(G: nx.Graph) -> nx.Graph:
    """提取图中的最大连通分量."""
    print("提取最大连通分量...")
    connected_components = list(nx.connected_components(G))
    if not connected_components:
        print("图为空，没有连通分量。")
        return nx.Graph()
    largest_component = max(connected_components, key=len)
    G_largest = G.subgraph(largest_component).copy()
    print(f"最大连通分量包含 {G_largest.number_of_nodes()} 个节点和 {G_largest.number_of_edges()} 条边。")
    return G_largest

def analyze_basic_stats(G: nx.Graph) -> Dict:
    """分析网络的基本统计信息."""
    print("计算基本网络统计信息...")
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    avg_degree = np.mean([d for _, d in G.degree()])
    density = nx.density(G)
    num_components = nx.number_connected_components(G)
    print(f"节点数量: {nodes}")
    print(f"边的数量: {edges}")
    print(f"平均度: {avg_degree:.2f}")
    print(f"图密度: {density:.4f}")
    print(f"连通分量数量: {num_components}")
    return {
        'nodes': nodes,
        'edges': edges,
        'avg_degree': avg_degree,
        'density': density,
        'num_components': num_components
    }

def analyze_shortest_paths(G: nx.Graph, graph_name: str = "") -> Dict:
    """分析连通图的最短路径长度."""
    num_components = nx.number_connected_components(G)
    if num_components == 1:
        print("计算最短路径长度和直径...")
        shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        diameter = max(nx.eccentricity(G, sp=shortest_path_lengths).values())
        avg_shortest_path = np.mean([np.mean(list(spl.values())) for spl in shortest_path_lengths.values()])

        path_lengths = np.zeros(diameter + 1, dtype=int)
        for pls in shortest_path_lengths.values():
            pl, cnts = np.unique(list(pls.values()), return_counts=True)
            path_lengths[pl] += cnts

        freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()

        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(1, diameter + 1), height=freq_percent)
        plt.title("Distribution of Shortest Path Length")
        plt.xlabel("Shortest Path Length")
        plt.ylabel("Frequency (%)")
        plt.savefig(f"result/{graph_name}_shortest_path_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()  

        path_length_stats = {
            'min': min(pl for pls in shortest_path_lengths.values() for pl in pls.values()),
            'max': diameter,
            'mean': avg_shortest_path,
            'median': np.median([pl for pls in shortest_path_lengths.values() for pl in pls.values()])
        }
        print(f"直径: {diameter}")
        print(f"平均最短路径长度: {avg_shortest_path:.4f}")
        print(f"路径长度统计: {path_length_stats}")
        return {'diameter': diameter, 'avg_shortest_path': avg_shortest_path, 'path_length_stats': path_length_stats}
    else:
        print("图不连通，跳过最短路径长度和直径的计算。")
        return {}

def analyze_communities(G: nx.Graph) -> Tuple[List, float]:
    """分析网络的社区结构."""
    print("分析社区结构...")
    communities = community.greedy_modularity_communities(G)
    modularity = community.modularity(G, communities)
    print(f"社区数量: {len(communities)}")
    print(f"模块化系数: {modularity:.4f}")
    return communities, modularity

def visualize_community_structure(G: nx.Graph, communities: List[Set], graph_name: str):
    """可视化并保存社区结构"""
    title = f"{graph_name} Community Structure"
    print(f"可视化{title}的社区结构...")
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    degree_centrality = nx.degree_centrality(G)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos,
                            node_color=[community_map.get(node, -1) for node in G.nodes()],
                            node_size=[v * 500 for v in degree_centrality.values()],
                            cmap=plt.colormaps['viridis'])
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)

    for i in range(len(communities)):
        community_nodes = [node for node, comm in community_map.items() if comm == i]
        if community_nodes:
            central_node = max(community_nodes, key=lambda node: degree_centrality[node])
            nx.draw_networkx_labels(G, pos,
                                     labels={central_node: central_node},
                                     font_size=10,
                                     font_color='black')

    plt.title(title)
    plt.savefig(f"result/{graph_name}_community_structure.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def analyze_robustness(G: nx.Graph, graph_name: str = "") -> Dict:
    """分析网络的鲁棒性."""
    print("分析网络鲁棒性...")
    num_nodes = len(G)
    if num_nodes == 0:
        return {'original_components': 0, 'targeted_attack_components': 0, 'random_attack_components': 0, 'robustness_ratio': 0}

    num_remove = int(0.05 * num_nodes)

    # 针对性攻击
    degrees = dict(G.degree())
    sorted_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
    high_degree_nodes = [node for node, _ in sorted_degree[:num_remove]]
    G_targeted = G.copy()
    G_targeted.remove_nodes_from(high_degree_nodes)
    targeted_components = nx.number_connected_components(G_targeted)

    # 随机攻击
    random_nodes = random.sample(list(G.nodes()), num_remove)
    G_random = G.copy()
    G_random.remove_nodes_from(random_nodes)
    random_components = nx.number_connected_components(G_random)

    original_components = nx.number_connected_components(G)
    robustness_ratio = (targeted_components - random_components) / original_components if original_components > 0 else 0

    print(f"原始图的连通分量数量: {original_components}")
    print(f"针对性攻击后的连通分量数量: {targeted_components}")
    print(f"随机攻击后的连通分量数量: {random_components}")
    print(f"网络鲁棒性比例: {robustness_ratio:.4f}")

    plt.figure(figsize=(8, 5))
    plt.bar(['Original', 'Targeted Attack', 'Random Attack'],
            [original_components, targeted_components, random_components],
            color=['blue', 'red', 'green'])
    plt.title('Network Robustness Analysis')
    plt.ylabel('Number of Connected Components')
    plt.savefig(f"result/{graph_name}_robustness.png", dpi=300)
    plt.show()
    plt.close()  # 新增关闭

    return {
        'original_components': original_components,
        'targeted_attack_components': targeted_components,
        'random_attack_components': random_components,
        'robustness_ratio': robustness_ratio
    }

def analyze_clustering(G: nx.Graph) -> Dict:
    """分析网络的聚类系数."""
    print("分析聚类系数...")
    global_clustering = nx.transitivity(G)
    avg_local_clustering = nx.average_clustering(G)
    print(f"全局聚类系数: {global_clustering:.4f}")
    print(f"平均局部聚类系数: {avg_local_clustering:.4f}")
    return {'global_clustering': global_clustering, 'avg_local_clustering': avg_local_clustering}

def analyze_small_world(G: nx.Graph) -> Dict:
    """分析网络的小世界特性."""
    print("分析小世界特性...")
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_edges > 0 and n_nodes > 1:
        degree_sequence = [d for n, d in G.degree()]
        G_rand = nx.configuration_model(degree_sequence, create_using=nx.Graph())
        G_rand = nx.Graph(G_rand)  # 移除平行边和自环

        C = nx.average_clustering(G)
        C_rand = nx.average_clustering(G_rand) if G_rand.number_of_edges() > 0 else 0

        L = nx.average_shortest_path_length(G) if nx.is_connected(G) and n_nodes > 1 else float('inf')
        L_rand = nx.average_shortest_path_length(G_rand) if nx.is_connected(G_rand) and G_rand.number_of_nodes() > 1 and C_rand > 0 else float('inf')

        sigma = (C / C_rand) / (L / L_rand) if C_rand > 0 and L_rand != float('inf') and L != float('inf') else float('inf')

        print(f"小世界 Sigma 值: {sigma:.4f}（若 >> 1，则为小世界网络）")
        return {'sigma': sigma}
    else:
        print("边或节点数量不足，无法计算小世界 Sigma 值。")
        return {'sigma': float('inf')}

def analyze_centralities(G: nx.Graph, graph_name: str = "", top_n: int = 20) -> Dict:
    """分析网络的中心性指标并可视化."""
    print("分析并可视化中心性...")
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, k=min(1000, G.number_of_nodes())) # 使用 k 进行近似计算，避免计算量过大
    closeness_centrality = nx.closeness_centrality(G)

    top_degree_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_betweenness_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_closeness_nodes = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.bar(range(len(top_degree_nodes)), [val for _, val in top_degree_nodes], color='orange')
    plt.title('Top Degree Centrality')
    plt.xticks(range(len(top_degree_nodes)), [key for key, _ in top_degree_nodes], rotation=90, fontsize=8)
    plt.ylabel('Centrality Score')

    plt.subplot(1, 3, 2)
    plt.bar(range(len(top_betweenness_nodes)), [val for _, val in top_betweenness_nodes], color='green')
    plt.title('Top Betweenness Centrality')
    plt.xticks(range(len(top_betweenness_nodes)), [key for key, _ in top_betweenness_nodes], rotation=90, fontsize=8)
    plt.ylabel('Centrality Score')

    plt.subplot(1, 3, 3)
    plt.bar(range(len(top_closeness_nodes)), [val for _, val in top_closeness_nodes], color='blue')
    plt.title('Top Closeness Centrality')
    plt.xticks(range(len(top_closeness_nodes)), [key for key, _ in top_closeness_nodes], rotation=90, fontsize=8)
    plt.ylabel('Centrality Score')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(degree_centrality.values(), bins=20, color='orange')
    plt.title('Degree Centrality Distribution')
    plt.xlabel('Degree Centrality')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(betweenness_centrality.values(), bins=20, color='green')
    plt.title('Betweenness Centrality Distribution')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(closeness_centrality.values(), bins=20, color='blue')
    plt.title('Closeness Centrality Distribution')
    plt.xlabel('Closeness Centrality')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"result/{graph_name}_centrality_distributions.png", dpi=300)
    plt.show()
    plt.close()  

    return {
        'degree': degree_centrality,
        'betweenness': betweenness_centrality,
        'closeness': closeness_centrality,
        'top_degree': top_degree_nodes,
        'top_betweenness': top_betweenness_nodes,
        'top_closeness': top_closeness_nodes
    }

def visualize_subgraph(G: nx.Graph, key_nodes: List, title: str, filename: str, layout: str = 'spring'):
    """可视化关键节点周围的子图."""
    subgraph_nodes = set()
    for node in key_nodes:
        subgraph_nodes.add(node)
        subgraph_nodes.update(G.neighbors(node))
    subgraph = G.subgraph(subgraph_nodes)

    if layout == 'spring':
        pos = nx.spring_layout(subgraph, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(subgraph)
    elif layout == 'random':
        pos = nx.random_layout(subgraph, seed=42)
    else:
        pos = nx.spring_layout(subgraph, seed=42)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(subgraph, pos, nodelist=key_nodes, node_color='red', node_size=200, label='Key Nodes')
    nx.draw_networkx_nodes(subgraph, pos, nodelist=set(subgraph.nodes()) - set(key_nodes), node_color='lightblue', node_size=100, label='Other Nodes')
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def visualize_network(G: nx.Graph, title: str = "", sample_size: int = 1000) -> None:
    """可视化整个网络（可选择采样）. """
    print(f"可视化{title}...")
    if G.number_of_nodes() > sample_size:
        nodes = random.sample(list(G.nodes()), sample_size)
        G_sampled = G.subgraph(nodes)
        pos = nx.spring_layout(G_sampled, seed=42)
        nx.draw(G_sampled, pos, node_size=10, with_labels=False, edge_color='gray', alpha=0.7)
    else:
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=10, with_labels=False, edge_color='gray', alpha=0.7)
    plt.title(title)
    plt.show()

def visualize_key_nodes(G: nx.Graph, top_degree: List[Tuple], top_betweenness: List[Tuple], top_closeness: List[Tuple], title: str = ""):
    """可视化关键中心性节点周围的子图."""
    print(f"可视化{title}的关键节点子图...")
    top_degree_nodes = [node for node, _ in top_degree]
    visualize_subgraph(G, top_degree_nodes[:5], f"{title} - Top 5 Degree Centrality Nodes", f"{title}_top_degree_subgraph.png", layout='spring')

    top_betweenness_nodes = [node for node, _ in top_betweenness]
    visualize_subgraph(G, top_betweenness_nodes[:5], f"{title} - Top 5 Betweenness Centrality Nodes", f"{title}_top_betweenness_subgraph.png", layout='circular')

    top_closeness_nodes = [node for node, _ in top_closeness]
    visualize_subgraph(G, top_closeness_nodes[:5], f"{title} - Top 5 Closeness Centrality Nodes", f"{title}_top_closeness_subgraph.png", layout='random')

def perform_network_analysis(G: nx.Graph, graph_name: str = "Network") -> Dict:
    """对给定的图执行完整的网络分析"""
    print(f"\n--- 分析 {graph_name} ---")
    
    basic_stats = analyze_basic_stats(G)
    shortest_path_stats = analyze_shortest_paths(G, graph_name)  # 传递graph_name
    communities, modularity = analyze_communities(G)
    robustness_metrics = analyze_robustness(G, graph_name)  # 传递graph_name
    clustering_coeffs = analyze_clustering(G)
    small_world_coeffs = analyze_small_world(G)
    centrality_measures = analyze_centralities(G, graph_name)  # 传递graph_name
    # 修复参数传递方式
    visualize_community_structure(G, communities, graph_name)
    visualize_network(G, title=graph_name)
    visualize_key_nodes(G, centrality_measures['top_degree'], 
                       centrality_measures['top_betweenness'],
                       centrality_measures['top_closeness'],
                       graph_name)

    return {
        'basic_stats': basic_stats,
        'shortest_path_stats': shortest_path_stats,
        'communities': communities,
        'modularity': modularity,
        'robustness': robustness_metrics,
        'clustering': clustering_coeffs,
        'small_world': small_world_coeffs,
        'centralities': centrality_measures
    }

def compare_analysis_results(original: Dict, largest: Dict):
    """对比分析结果"""
    print("\n--- 原始图 vs 最大连通图 分析结果比较 ---")
    
    comparisons = {
        '基本统计': (original['basic_stats'], largest['basic_stats']),
        '模块化系数': (original['modularity'], largest['modularity']),
        '鲁棒性指标': (original['robustness'], largest['robustness']),
        '最短路径统计': (original['shortest_path_stats'], largest['shortest_path_stats']),
        '聚类系数': (original['clustering'], largest['clustering']),
        '小世界特性': (original['small_world'], largest['small_world'])
    }
    
    for title, (orig, lrg) in comparisons.items():
        print(f"\n{title}:")
        print(f"  原始图: {orig}")
        print(f"  最大连通图: {lrg}")
    
    print("\n中心性指标对比:")
    for metric in ['top_degree', 'top_betweenness', 'top_closeness']:
        print(f"  {metric}:")
        print(f"    原始图前5: {original['centralities'][metric][:5]}")
        print(f"    最大连通图前5: {largest['centralities'][metric][:5]}")

# -------------------- 主程序 --------------------

if __name__ == "__main__":
    # 创建结果目录
    os.makedirs('result', exist_ok=True)
    
    # 步骤 1：加载原始图数据
    filepath = 'data/ca-GrQc.txt.gz'
    original_graph = load_graph_from_edgelist(filepath)

    # 步骤 2：提取最大连通分量
    largest_connected_component = get_largest_connected_component(original_graph)

    # 步骤 3：对原始图进行网络分析
    original_analysis_results = perform_network_analysis(original_graph, "原始图")

    # 步骤 4：对最大连通分量进行网络分析
    largest_component_analysis_results = perform_network_analysis(largest_connected_component, "最大连通图")

    # 步骤 5：比较分析结果
    compare_analysis_results(original_analysis_results, largest_component_analysis_results)



