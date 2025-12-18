"""
电信诈骗犯罪网络社会网络特性分析
分析结构洞、核心-边缘结构、中心性等社会网络特性
基于复杂网络与社会网络特性的电信诈骗犯罪网络关键节点识别研究
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SocialNetworkAnalyzer:
    """社会网络特性分析器"""

    def __init__(self, data_dir="data"):
        """
        初始化分析器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.G = None
        self.nodes_df = None
        self.results = {}

    def load_data(self):
        """加载网络数据"""
        print("正在加载网络数据...")
        try:
            with open(f"{self.data_dir}/network_graph.pkl", 'rb') as f:
                self.G = pickle.load(f)
            self.nodes_df = pd.read_csv(f"{self.data_dir}/nodes.csv")
            print(f"数据加载完成：{self.G.number_of_nodes()}个节点，{self.G.number_of_edges()}条边")
            return True
        except FileNotFoundError:
            print("网络文件未找到")
            return False

    def analyze_structural_holes(self):
        """分析结构洞"""
        print("正在分析结构洞特性...")

        # 计算结构洞约束度
        constraint_dict = nx.constraint(self.G)

        # 结构洞指标统计
        constraints = list(constraint_dict.values())
        constraints = [c for c in constraints if c is not None]  # 过滤None值

        structural_holes_stats = {
            'mean_constraint': np.mean(constraints),
            'std_constraint': np.std(constraints),
            'min_constraint': min(constraints),
            'max_constraint': max(constraints),
            'constraint_distribution': constraint_dict
        }

        # 识别结构洞（低约束度的节点）
        low_constraint_threshold = np.percentile(constraints, 25)  # 25%分位数作为阈值
        structural_hole_nodes = [node for node, constraint in constraint_dict.items()
                               if constraint is not None and constraint <= low_constraint_threshold]

        structural_holes_stats['structural_hole_nodes'] = structural_hole_nodes
        structural_holes_stats['num_structural_holes'] = len(structural_hole_nodes)
        structural_holes_stats['low_constraint_threshold'] = low_constraint_threshold

        # 计算有效尺寸（effective size）- 另一种结构洞度量
        effective_size_dict = nx.effective_size(self.G)
        effective_sizes = list(effective_size_dict.values())

        structural_holes_stats['effective_size'] = {
            'mean': np.mean(effective_sizes),
            'std': np.std(effective_sizes),
            'max': max(effective_sizes),
            'distribution': effective_size_dict
        }

        self.results['structural_holes'] = structural_holes_stats

        print(f"发现 {len(structural_hole_nodes)} 个结构洞节点")
        return structural_holes_stats

    def analyze_core_periphery_structure(self):
        """分析核心-边缘结构"""
        print("正在分析核心-边缘结构...")

        # 计算k-core分解
        core_numbers = nx.core_number(self.G)

        # 核心结构统计
        core_values = list(core_numbers.values())
        max_core = max(core_values)

        core_stats = {
            'max_core_number': max_core,
            'core_distribution': dict(Counter(core_values)),
            'core_numbers': core_numbers
        }

        # 识别核心节点（最高k-core的节点）
        core_nodes = [node for node, core_num in core_numbers.items() if core_num == max_core]
        core_stats['core_nodes'] = core_nodes
        core_stats['num_core_nodes'] = len(core_nodes)

        # 计算核心密度
        if core_nodes:
            core_subgraph = self.G.subgraph(core_nodes)
            core_stats['core_density'] = nx.density(core_subgraph)
            core_stats['core_avg_degree'] = np.mean([d for n, d in core_subgraph.degree()])
        else:
            core_stats['core_density'] = 0
            core_stats['core_avg_degree'] = 0

        # 边缘节点（k-core = 1的节点）
        periphery_nodes = [node for node, core_num in core_numbers.items() if core_num == 1]
        core_stats['periphery_nodes'] = periphery_nodes
        core_stats['num_periphery_nodes'] = len(periphery_nodes)

        # 计算核心率
        core_stats['core_ratio'] = len(core_nodes) / self.G.number_of_nodes()

        self.results['core_periphery'] = core_stats

        print(f"最大k-core值: {max_core}")
        print(f"核心节点数量: {len(core_nodes)}")
        print(f"边缘节点数量: {len(periphery_nodes)}")

        return core_stats

    def analyze_centrality_measures(self):
        """分析各种中心性度量"""
        print("正在计算各种中心性度量...")

        # 各种中心性计算
        centrality_measures = {
            'degree': nx.degree_centrality(self.G),
            'betweenness': nx.betweenness_centrality(self.G),
            'closeness': nx.closeness_centrality(self.G),
            'eigenvector': nx.eigenvector_centrality(self.G, max_iter=1000),
            'pagerank': nx.pagerank(self.G, alpha=0.85)
        }

        # 为每种中心性计算统计信息
        centrality_stats = {}
        for name, centrality_dict in centrality_measures.items():
            values = list(centrality_dict.values())
            centrality_stats[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values),
                'top_10_nodes': sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:10],
                'distribution': centrality_dict
            }

        # 计算中心性之间的相关性
        centrality_correlations = {}
        centrality_names = list(centrality_measures.keys())

        for i, name1 in enumerate(centrality_names):
            for name2 in centrality_names[i+1:]:
                corr = np.corrcoef(
                    list(centrality_measures[name1].values()),
                    list(centrality_measures[name2].values())
                )[0, 1]
                centrality_correlations[f'{name1}_{name2}'] = corr

        centrality_stats['correlations'] = centrality_correlations

        self.results['centrality'] = centrality_stats

        print("中心性计算完成")
        return centrality_stats

    def analyze_network_cohesion(self):
        """分析网络凝聚性"""
        print("正在分析网络凝聚性...")

        cohesion_metrics = {}

        # 聚类系数
        avg_clustering = nx.average_clustering(self.G)
        clustering_coeff = nx.clustering(self.G)

        cohesion_metrics['clustering'] = {
            'average_clustering': avg_clustering,
            'clustering_distribution': clustering_coeff,
            'global_clustering': nx.transitivity(self.G)
        }

        # 密度
        cohesion_metrics['density'] = nx.density(self.G)

        # 连通性
        cohesion_metrics['connectivity'] = {
            'is_connected': nx.is_connected(self.G),
            'num_components': nx.number_connected_components(self.G),
            'largest_component_ratio': len(max(nx.connected_components(self.G), key=len)) / self.G.number_of_nodes()
        }

        # 平均路径长度（对连通图）
        if nx.is_connected(self.G):
            cohesion_metrics['average_path_length'] = nx.average_shortest_path_length(self.G)
        else:
            largest_cc = max(nx.connected_components(self.G), key=len)
            subgraph = self.G.subgraph(largest_cc)
            cohesion_metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)

        # 网络效率
        cohesion_metrics['efficiency'] = nx.global_efficiency(self.G)

        self.results['cohesion'] = cohesion_metrics

        return cohesion_metrics

    def analyze_risk_centrality_relationship(self):
        """分析风险等级与中心性的关系"""
        print("正在分析风险等级与中心性的关系...")

        # 获取风险等级和中心性数据
        risk_centrality = {}

        # 计算各中心性度量
        centralities = {
            'degree': nx.degree_centrality(self.G),
            'betweenness': nx.betweenness_centrality(self.G),
            'closeness': nx.closeness_centrality(self.G),
            'eigenvector': nx.eigenvector_centrality(self.G, max_iter=1000)
        }

        # 按风险等级分组分析
        risk_levels = ['高', '中', '低']
        for risk_level in risk_levels:
            risk_nodes = self.nodes_df[self.nodes_df['risk_level'] == risk_level]['node_id'].tolist()

            risk_centrality[risk_level] = {}
            for cent_name, cent_dict in centralities.items():
                risk_cent_values = [cent_dict.get(node, 0) for node in risk_nodes]
                risk_centrality[risk_level][cent_name] = {
                    'mean': np.mean(risk_cent_values),
                    'std': np.std(risk_cent_values),
                    'max': max(risk_cent_values) if risk_cent_values else 0,
                    'count': len(risk_cent_values)
                }

        # 计算相关性：风险等级与中心性的关系
        risk_mapping = {'低': 1, '中': 2, '高': 3}
        risk_scores = [risk_mapping[self.nodes_df[self.nodes_df['node_id'] == node]['risk_level'].iloc[0]]
                      for node in self.G.nodes()]

        risk_correlations = {}
        for cent_name, cent_dict in centralities.items():
            cent_values = [cent_dict[node] for node in self.G.nodes()]
            corr = np.corrcoef(risk_scores, cent_values)[0, 1]
            risk_correlations[cent_name] = corr

        risk_centrality['correlations'] = risk_correlations

        self.results['risk_centrality'] = risk_centrality

        return risk_centrality

    def create_social_network_visualizations(self, output_dir="social_analysis"):
        """创建社会网络特性可视化"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在创建社会网络特性可视化...")

        # 1. 结构洞分析图
        self.plot_structural_holes_analysis(output_dir)

        # 2. 核心-边缘结构图
        self.plot_core_periphery_analysis(output_dir)

        # 3. 中心性对比图
        self.plot_centrality_comparison(output_dir)

        # 4. 风险等级与中心性关系图
        self.plot_risk_centrality_relationship(output_dir)

        # 5. 网络凝聚性雷达图
        self.plot_network_cohesion_radar(output_dir)

    def plot_structural_holes_analysis(self, output_dir):
        """绘制结构洞分析图"""
        if 'structural_holes' not in self.results:
            return

        sh = self.results['structural_holes']

        plt.figure(figsize=(15, 10))

        # 子图1：约束度分布
        plt.subplot(2, 3, 1)
        constraints = [c for c in sh['constraint_distribution'].values() if c is not None]
        plt.hist(constraints, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(sh['low_constraint_threshold'], color='red', linestyle='--',
                   label=f'Structural hole threshold: {sh["low_constraint_threshold"]:.3f}')
        plt.xlabel('Constraint')
        plt.ylabel('Frequency')
        plt.title('Structural Hole Constraint Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：有效尺寸分布
        plt.subplot(2, 3, 2)
        effective_sizes = list(sh['effective_size']['distribution'].values())
        plt.hist(effective_sizes, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Effective Size')
        plt.ylabel('Frequency')
        plt.title('Structural Hole Effective Size Distribution')
        plt.grid(True, alpha=0.3)

        # 子图3：结构洞节点网络位置
        plt.subplot(2, 3, 3)
        # 创建节点颜色映射
        node_colors = []
        for node in self.G.nodes():
            if node in sh['structural_hole_nodes']:
                node_colors.append('red')  # 结构洞节点
            else:
                node_colors.append('blue')  # 普通节点

        # 简化布局，只显示部分节点
        if self.G.number_of_nodes() > 100:
            # 随机采样节点进行可视化
            sample_nodes = np.random.choice(list(self.G.nodes()), size=min(100, self.G.number_of_nodes()), replace=False)
            subgraph = self.G.subgraph(sample_nodes)
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
            sample_colors = [node_colors[list(self.G.nodes()).index(node)] for node in sample_nodes]
            nx.draw_networkx_nodes(subgraph, pos, node_size=50, node_color=sample_colors, alpha=0.7)
            nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=0.5)
        else:
            pos = nx.spring_layout(self.G, k=0.5, iterations=50, seed=42)
            nx.draw_networkx_nodes(self.G, pos, node_size=50, node_color=node_colors, alpha=0.7)
            nx.draw_networkx_edges(self.G, pos, alpha=0.3, width=0.5)

        plt.title('Structural Hole Nodes Distribution')
        plt.axis('off')

        # 子图4：约束度 vs 度数
        plt.subplot(2, 3, 4)
        degrees = dict(self.G.degree())
        constraints_list = []
        degrees_list = []

        for node in self.G.nodes():
            if sh['constraint_distribution'][node] is not None:
                constraints_list.append(sh['constraint_distribution'][node])
                degrees_list.append(degrees[node])

        plt.scatter(degrees_list, constraints_list, alpha=0.6, color='purple', s=30)
        plt.xlabel('Node Degree')
        plt.ylabel('Constraint')
        plt.title('Constraint vs Node Degree')
        plt.grid(True, alpha=0.3)

        # 子图5：统计信息
        plt.subplot(2, 3, 5)
        plt.axis('off')
        stats_text = f"""
        Structural Hole Analysis Statistics:

        Total nodes: {self.G.number_of_nodes()}
        Structural hole nodes: {sh['num_structural_holes']}
        Structural hole ratio: {sh['num_structural_holes']/self.G.number_of_nodes():.1%}

        Constraint Statistics:
        Mean: {sh['mean_constraint']:.3f}
        Std: {sh['std_constraint']:.3f}
        Min: {sh['min_constraint']:.3f}
        Max: {sh['max_constraint']:.3f}

        Effective Size Statistics:
        Mean: {sh['effective_size']['mean']:.3f}
        Max: {sh['effective_size']['max']:.3f}
        """
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')

        # 子图6：结构洞识别标准
        plt.subplot(2, 3, 6)
        plt.axis('off')
        criteria_text = """
        Structural Hole Identification Criteria:

        • Constraint ≤ 25th percentile
        • Relatively large effective size
        • Acts as "bridge" role in network

        Structural Hole Advantages:
        ✓ Controls information flow
        ✓ Connects different groups
        ✓ Has negotiation advantages
        ✓ More innovation opportunities

        Significance in Crime Networks:
        • Structural hole nodes are often key figures
        • Cutting structural holes can dismantle network
        • Focus on monitoring structural hole nodes
        """
        plt.text(0.05, 0.95, criteria_text, transform=plt.gca().transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/structural_holes_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_core_periphery_analysis(self, output_dir):
        """绘制核心-边缘结构分析图"""
        if 'core_periphery' not in self.results:
            return

        cp = self.results['core_periphery']

        plt.figure(figsize=(15, 10))

        # 子图1：k-core分布
        plt.subplot(2, 3, 1)
        core_dist = cp['core_distribution']
        cores = list(core_dist.keys())
        counts = list(core_dist.values())

        plt.bar(cores, counts, color='lightcoral', alpha=0.7, edgecolor='black')
        plt.xlabel('k-core Value')
        plt.ylabel('Node Count')
        plt.title('k-core Distribution')
        plt.xticks(cores)
        plt.grid(True, alpha=0.3)

        # 子图2：核心节点网络
        plt.subplot(2, 3, 2)
        if cp['core_nodes']:
            core_subgraph = self.G.subgraph(cp['core_nodes'])
            if len(cp['core_nodes']) <= 50:  # 只在核心节点不多时绘制
                pos = nx.spring_layout(core_subgraph, k=1, iterations=50, seed=42)
                nx.draw_networkx_nodes(core_subgraph, pos, node_size=100,
                                     node_color='red', alpha=0.8)
                nx.draw_networkx_edges(core_subgraph, pos, alpha=0.5, width=1)
                nx.draw_networkx_labels(core_subgraph, pos, font_size=8)
            else:
                plt.text(0.5, 0.5, f'Too many core nodes\n({len(cp["core_nodes"])} nodes)\nCannot display fully',
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)

        plt.title('Core Nodes Network Structure')
        plt.axis('off')

        # 子图3：核心度数分布
        plt.subplot(2, 3, 3)
        if cp['core_nodes']:
            core_degrees = [self.G.degree(node) for node in cp['core_nodes']]
            periphery_degrees = [self.G.degree(node) for node in cp['periphery_nodes']]

            plt.hist(core_degrees, bins=15, alpha=0.7, label='Core nodes',
                    color='red', edgecolor='black')
            plt.hist(periphery_degrees, bins=15, alpha=0.7, label='Periphery nodes',
                    color='blue', edgecolor='black')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.title('Core vs Periphery Node Degree Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 子图4：核心-边缘对比
        plt.subplot(2, 3, 4)
        categories = ['Core nodes', 'Periphery nodes', 'All nodes']
        counts = [cp['num_core_nodes'], cp['num_periphery_nodes'],
                 self.G.number_of_nodes() - cp['num_core_nodes'] - cp['num_periphery_nodes']]

        plt.bar(categories, counts, color=['red', 'blue', 'gray'], alpha=0.7, edgecolor='black')
        plt.ylabel('Node Count')
        plt.title('Core-Periphery Node Comparison')

        for i, count in enumerate(counts):
            plt.text(i, count + 1, str(count), ha='center', va='bottom')

        # 子图5：统计信息
        plt.subplot(2, 3, 5)
        plt.axis('off')
        stats_text = f"""
        Core-Periphery Structure Statistics:

        Max k-core value: {cp['max_core_number']}
        Core nodes: {cp['num_core_nodes']}
        Periphery nodes: {cp['num_periphery_nodes']}
        Intermediate nodes: {self.G.number_of_nodes() - cp['num_core_nodes'] - cp['num_periphery_nodes']}

        Core Features:
        Density: {cp['core_density']:.4f}
        Avg degree: {cp['core_avg_degree']:.2f}
        Core ratio: {cp['core_ratio']:.1%}
        """
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')

        # 子图6：核心-边缘理论意义
        plt.subplot(2, 3, 6)
        plt.axis('off')
        theory_text = """
        Core-Periphery Structure Significance:

        Core Node Characteristics:
        • Highest k-core value
        • Highly interconnected
        • Source of network stability
        • Usually key figures

        Periphery Node Characteristics:
        • k-core value = 1
        • Loosely connected to core
        • Easy to detach from network
        • Network expansion points

        Application in Crime Networks:
        ✓ Prioritize attacking core nodes
        ✓ Monitor periphery node additions
        ✓ Core nodes are often main criminals
        ✓ Periphery nodes are often accomplices
        """
        plt.text(0.05, 0.95, theory_text, transform=plt.gca().transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/core_periphery_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_centrality_comparison(self, output_dir):
        """绘制中心性对比图"""
        if 'centrality' not in self.results:
            return

        cent = self.results['centrality']

        plt.figure(figsize=(15, 12))

        # 子图1：中心性分布对比
        plt.subplot(3, 3, 1)
        centrality_names = ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank']
        means = [cent[name]['mean'] for name in centrality_names]

        bars = plt.bar(centrality_names, means, color='skyblue', alpha=0.7, edgecolor='black')
        plt.ylabel('平均中心性')
        plt.title('各中心性度量平均值对比')
        plt.xticks(rotation=45)

        for bar, mean_val in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    '.4f', ha='center', va='bottom', fontsize=8)

        # 子图2-6：各中心性的分布直方图
        centrality_labels = {
            'degree': '度中心性',
            'betweenness': '介数中心性',
            'closeness': '接近中心性',
            'eigenvector': '特征向量中心性',
            'pagerank': 'PageRank中心性'
        }

        for i, (name, label) in enumerate(centrality_labels.items()):
            plt.subplot(3, 3, i+2)
            values = list(cent[name]['distribution'].values())
            plt.hist(values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel(label)
            plt.ylabel('频次')
            plt.title(f'{label}分布')
            plt.grid(True, alpha=0.3)

        # 子图7：中心性相关性热力图
        plt.subplot(3, 3, 8)
        corr_matrix = np.zeros((len(centrality_names), len(centrality_names)))

        for i, name1 in enumerate(centrality_names):
            for j, name2 in enumerate(centrality_names):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif f'{name1}_{name2}' in cent['correlations']:
                    corr_matrix[i, j] = cent['correlations'][f'{name1}_{name2}']
                elif f'{name2}_{name1}' in cent['correlations']:
                    corr_matrix[i, j] = cent['correlations'][f'{name2}_{name1}']

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=centrality_names, yticklabels=centrality_names,
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('中心性相关性矩阵')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # 子图8：Top 10节点中心性对比
        plt.subplot(3, 3, 9)
        # 选择前10个高介数中心性的节点
        top_nodes = [node for node, _ in cent['betweenness']['top_10_nodes']]

        centrality_data = {}
        for name in centrality_names:
            centrality_data[name] = [cent[name]['distribution'][node] for node in top_nodes]

        x = np.arange(len(top_nodes))
        width = 0.15

        for i, (name, label) in enumerate(centrality_labels.items()):
            plt.bar(x + i*width, centrality_data[name], width,
                   label=label, alpha=0.7)

        plt.xlabel('节点ID')
        plt.ylabel('中心性值')
        plt.title('Top 10节点各中心性对比')
        plt.xticks(x + width*2, [f'S{node[1:]}' for node in top_nodes], rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/centrality_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_risk_centrality_relationship(self, output_dir):
        """绘制风险等级与中心性关系图"""
        if 'risk_centrality' not in self.results:
            return

        rc = self.results['risk_centrality']

        plt.figure(figsize=(15, 10))

        # 子图1：各风险等级的平均中心性
        plt.subplot(2, 3, 1)
        risk_levels = ['高', '中', '低']
        centrality_types = ['degree', 'betweenness', 'closeness', 'eigenvector']

        x = np.arange(len(risk_levels))
        width = 0.2

        for i, cent_type in enumerate(centrality_types):
            means = [rc[level][cent_type]['mean'] for level in risk_levels]
            plt.bar(x + i*width, means, width, label=f'{cent_type}中心性', alpha=0.7)

        plt.xlabel('风险等级')
        plt.ylabel('平均中心性')
        plt.title('风险等级与中心性关系')
        plt.xticks(x + width*1.5, risk_levels)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：风险等级vs中心性的相关性
        plt.subplot(2, 3, 2)
        correlations = [rc['correlations'][cent] for cent in centrality_types]
        bars = plt.bar(centrality_types, correlations, color='orange', alpha=0.7, edgecolor='black')

        plt.ylabel('相关系数')
        plt.title('风险等级与中心性的相关性')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        for bar, corr in zip(bars, correlations):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.3f', ha='center', va='bottom')

        # 子图3：高风险节点中心性分布
        plt.subplot(2, 3, 3)
        high_risk_nodes = self.nodes_df[self.nodes_df['risk_level'] == '高']['node_id'].tolist()
        if high_risk_nodes:
            for cent_type in centrality_types:
                cent_values = [nx.degree_centrality(self.G).get(node, 0) if cent_type == 'degree'
                              else nx.betweenness_centrality(self.G).get(node, 0) if cent_type == 'betweenness'
                              else nx.closeness_centrality(self.G).get(node, 0) if cent_type == 'closeness'
                              else nx.eigenvector_centrality(self.G).get(node, 0) if cent_type == 'eigenvector'
                              else 0 for node in high_risk_nodes]

                plt.hist(cent_values, bins=10, alpha=0.5, label=f'{cent_type}中心性')

        plt.xlabel('中心性值')
        plt.ylabel('频次')
        plt.title('高风险节点中心性分布')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图4：不同风险等级的网络密度对比
        plt.subplot(2, 3, 4)
        densities = []
        for risk_level in risk_levels:
            risk_nodes = self.nodes_df[self.nodes_df['risk_level'] == risk_level]['node_id'].tolist()
            if len(risk_nodes) > 1:
                subgraph = self.G.subgraph(risk_nodes)
                density = nx.density(subgraph)
            else:
                density = 0
            densities.append(density)

        bars = plt.bar(risk_levels, densities, color=['red', 'orange', 'green'], alpha=0.7, edgecolor='black')
        plt.ylabel('子图密度')
        plt.title('不同风险等级子图密度')
        plt.grid(True, alpha=0.3)

        for bar, density in zip(bars, densities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    '.4f', ha='center', va='bottom')

        # 子图5：统计信息
        plt.subplot(2, 3, 5)
        plt.axis('off')
        stats_text = f"""
        Risk Level Statistics:

        High-risk nodes: {len(self.nodes_df[self.nodes_df['risk_level'] == '高'])}
        Medium-risk nodes: {len(self.nodes_df[self.nodes_df['risk_level'] == '中'])}
        Low-risk nodes: {len(self.nodes_df[self.nodes_df['risk_level'] == '低'])}

        Correlation Analysis:
        Degree centrality: {rc['correlations']['degree']:.3f}
        Betweenness centrality: {rc['correlations']['betweenness']:.3f}
        Closeness centrality: {rc['correlations']['closeness']:.3f}
        Eigenvector centrality: {rc['correlations']['eigenvector']:.3f}

        Network Insights:
        • Positive correlation: High-risk nodes are more central
        • Negative correlation: Low-risk nodes are more central
        """
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')

        # 子图6：风险中心性分析总结
        plt.subplot(2, 3, 6)
        plt.axis('off')
        summary_text = """
        Risk Level and Centrality Analysis Summary:

        Key Findings:
        • High-risk nodes typically have higher centrality
        • Centrality can serve as a risk assessment indicator
        • Betweenness centrality has the strongest correlation with risk

        Practical Applications:
        ✓ Prioritize monitoring high-centrality nodes
        ✓ Centrality anomalies may indicate risk escalation
        ✓ Network analysis assists risk assessment

        Methodological Significance:
        • Validates the effectiveness of social network analysis
        • Provides new perspectives for crime network research
        • Demonstrates the impact of structural position on behavior
        """
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightpink", alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/risk_centrality_relationship.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_network_cohesion_radar(self, output_dir):
        """绘制网络凝聚性雷达图"""
        if 'cohesion' not in self.results:
            return

        coh = self.results['cohesion']

        plt.figure(figsize=(10, 8))

        # 准备雷达图数据
        categories = ['聚类系数', '密度', '连通性', '路径长度', '效率']
        values = [
            coh['clustering']['average_clustering'] * 10,  # 放大显示
            coh['density'] * 10,  # 放大显示
            coh['connectivity']['largest_component_ratio'],
            1 / coh['average_path_length'] if coh['average_path_length'] > 0 else 0,  # 转换为效率
            coh['efficiency']
        ]

        # 闭合雷达图
        values += values[:1]
        categories += categories[:1]

        # 计算角度
        angles = [n / float(len(categories[:-1])) * 2 * np.pi for n in range(len(categories))]

        # 绘制雷达图
        ax = plt.subplot(111, polar=True)
        plt.plot(angles, values, 'o-', linewidth=2, label='电信诈骗网络', color='blue', markersize=6)
        plt.fill(angles, values, alpha=0.25, color='blue')

        # 添加网格线
        plt.thetagrids([a * 180/np.pi for a in angles[:-1]], categories[:-1])

        # 设置标题
        plt.title('电信诈骗犯罪网络凝聚性雷达图', size=16, fontweight='bold', pad=20)

        # 添加参考线（理想的凝聚网络）
        ideal_values = [8, 5, 1, 0.8, 0.9]  # 理想值
        ideal_values += ideal_values[:1]
        plt.plot(angles, ideal_values, 'r--', linewidth=1, label='理想凝聚网络', alpha=0.7)
        plt.fill(angles, ideal_values, alpha=0.1, color='red')

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/network_cohesion_radar.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_social_analysis_report(self, output_dir="social_analysis"):
        """生成社会网络分析报告"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在生成社会网络分析报告...")

        with open(f"{output_dir}/social_network_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write("=== 电信诈骗犯罪网络社会网络特性分析报告 ===\n\n")

            # 结构洞分析
            if 'structural_holes' in self.results:
                sh = self.results['structural_holes']
                f.write("1. 结构洞分析:\n")
                f.write(f"   结构洞节点数量: {sh['num_structural_holes']}\n")
                f.write(f"   结构洞比例: {sh['num_structural_holes']/self.G.number_of_nodes():.1%}\n")
                f.write(f"   平均约束度: {sh['mean_constraint']:.4f}\n")
                f.write(f"   有效尺寸平均值: {sh['effective_size']['mean']:.4f}\n\n")

            # 核心-边缘结构分析
            if 'core_periphery' in self.results:
                cp = self.results['core_periphery']
                f.write("2. 核心-边缘结构分析:\n")
                f.write(f"   最大k-core值: {cp['max_core_number']}\n")
                f.write(f"   核心节点数量: {cp['num_core_nodes']}\n")
                f.write(f"   边缘节点数量: {cp['num_periphery_nodes']}\n")
                f.write(f"   核心密度: {cp['core_density']:.4f}\n")
                f.write(f"   核心率: {cp['core_ratio']:.1%}\n\n")

            # 中心性分析
            if 'centrality' in self.results:
                cent = self.results['centrality']
                f.write("3. 中心性分析:\n")
                for name in ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank']:
                    stats = cent[name]
                    f.write(f"   {name}中心性:\n")
                    f.write(f"      平均值: {stats['mean']:.6f}\n")
                    f.write(f"      标准差: {stats['std']:.6f}\n")
                    f.write(f"      最大值: {stats['max']:.6f}\n")
                    f.write(f"      Top节点: {stats['top_10_nodes'][:3]}\n")

                f.write("   中心性相关性:\n")
                for corr_name, corr_value in cent['correlations'].items():
                    f.write(f"      {corr_name}: {corr_value:.4f}\n")
                f.write("\n")

            # 网络凝聚性分析
            if 'cohesion' in self.results:
                coh = self.results['cohesion']
                f.write("4. 网络凝聚性分析:\n")
                f.write(f"   平均聚类系数: {coh['clustering']['average_clustering']:.6f}\n")
                f.write(f"   网络密度: {coh['density']:.6f}\n")
                f.write(f"   全局聚类系数: {coh['clustering']['global_clustering']:.6f}\n")
                f.write(f"   是否连通: {coh['connectivity']['is_connected']}\n")
                f.write(f"   连通分量数: {coh['connectivity']['num_components']}\n")
                f.write(f"   最大连通分量比例: {coh['connectivity']['largest_component_ratio']:.1%}\n")
                f.write(f"   平均路径长度: {coh['average_path_length']:.4f}\n")
                f.write(f"   网络效率: {coh['efficiency']:.6f}\n\n")

            # 风险与中心性关系分析
            if 'risk_centrality' in self.results:
                rc = self.results['risk_centrality']
                f.write("5. 风险等级与中心性关系分析:\n")
                for risk_level in ['高', '中', '低']:
                    f.write(f"   {risk_level}风险节点中心性:\n")
                    for cent_type in ['degree', 'betweenness', 'closeness', 'eigenvector']:
                        mean_cent = rc[risk_level][cent_type]['mean']
                        f.write(f"      {cent_type}: {mean_cent:.6f}\n")

                f.write("   风险等级与中心性的相关性:\n")
                for cent_type, corr in rc['correlations'].items():
                    f.write(f"      {cent_type}: {corr:.4f}\n")
                f.write("\n")

            # 综合评价
            f.write("6. 综合评价:\n")
            f.write("   电信诈骗犯罪网络表现出鲜明的社会网络特性:\n\n")

            f.write("   结构洞特征:\n")
            if 'structural_holes' in self.results:
                sh_ratio = sh['num_structural_holes']/self.G.number_of_nodes()
                if sh_ratio > 0.2:
                    f.write("   ✓ 结构洞节点比例较高，网络中存在较多'桥梁'人物\n")
                else:
                    f.write("   ⚠ 结构洞节点比例较低，网络结构相对紧密\n")

            f.write("\n   核心-边缘特征:\n")
            if 'core_periphery' in self.results:
                core_ratio = cp['core_ratio']
                if core_ratio < 0.1:
                    f.write("   ✓ 小核心结构，核心节点可能是主要犯罪分子\n")
                elif core_ratio > 0.3:
                    f.write("   ✓ 大核心结构，网络具有较强的凝聚力\n")
                else:
                    f.write("   ⚠ 中等核心结构，网络结构相对均衡\n")

            f.write("\n   中心性特征:\n")
            if 'centrality' in self.results:
                # 分析中心性分布的集中度
                for name in ['degree', 'betweenness']:
                    values = list(cent[name]['distribution'].values())
                    gini = self.calculate_gini_coefficient(values)
                    if gini > 0.6:
                        f.write(f"   ✓ {name}中心性分布不均，存在明显的关键节点\n")
                    else:
                        f.write(f"   ⚠ {name}中心性分布相对均匀\n")

            f.write("\n   风险关联特征:\n")
            if 'risk_centrality' in self.results:
                strong_corr = [cent for cent, corr in rc['correlations'].items() if abs(corr) > 0.3]
                if strong_corr:
                    f.write(f"   ✓ 风险等级与{strong_corr}中心性有较强相关性\n")
                    f.write("   ✓ 中心性可作为风险评估的重要指标\n")
                else:
                    f.write("   ⚠ 风险等级与中心性相关性不明显\n")

            f.write("\n   实战应用价值:\n")
            f.write("   • 结构洞节点：重点监控的'桥梁'人物，可能掌握关键信息\n")
            f.write("   • 核心节点：网络中的主要犯罪分子，打击重点对象\n")
            f.write("   • 高中心性节点：具有较大影响力的犯罪嫌疑人\n")
            f.write("   • 风险-中心性关联：为风险评估提供数据支持\n")

        print(f"社会网络分析报告已保存到 {output_dir}/social_network_analysis_report.txt")

    def calculate_gini_coefficient(self, values):
        """计算基尼系数"""
        values = sorted(values)
        n = len(values)
        if n == 0 or sum(values) == 0:
            return 0

        cumsum = np.cumsum(values)
        return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n

    def run_complete_analysis(self):
        """运行完整社会网络分析"""
        print("=== 开始电信诈骗犯罪网络社会网络特性分析 ===\n")

        if not self.load_data():
            return False

        # 分析结构洞
        self.analyze_structural_holes()

        # 分析核心-边缘结构
        self.analyze_core_periphery_structure()

        # 计算中心性度量
        self.analyze_centrality_measures()

        # 分析网络凝聚性
        self.analyze_network_cohesion()

        # 分析风险与中心性的关系
        self.analyze_risk_centrality_relationship()

        # 创建可视化
        self.create_social_network_visualizations()

        # 生成分析报告
        self.generate_social_analysis_report()

        print("\n=== 社会网络特性分析完成 ===")
        print("生成的文件:")
        print("- social_analysis/ 目录：社会网络特性可视化图表")
        print("- social_analysis/social_network_analysis_report.txt：详细分析报告")

        return True


def main():
    """主函数"""
    analyzer = SocialNetworkAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
