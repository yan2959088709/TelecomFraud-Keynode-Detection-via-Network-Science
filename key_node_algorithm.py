"""
电信诈骗犯罪网络关键节点识别算法优化
实现加权介数中心性算法和多指标融合评分模型
基于复杂网络与社会网络特性的电信诈骗犯罪网络关键节点识别研究
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ndcg_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class KeyNodeIdentifier:
    """关键节点识别器"""

    def __init__(self, data_dir="data"):
        """
        初始化关键节点识别器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.G = None
        self.nodes_df = None
        self.edges_df = None

        # 算法结果存储
        self.key_node_rankings = {}
        self.algorithm_performance = {}

    def load_data(self):
        """加载网络数据"""
        print("正在加载网络数据...")
        try:
            with open(f"{self.data_dir}/network_graph.pkl", 'rb') as f:
                self.G = pickle.load(f)
            self.nodes_df = pd.read_csv(f"{self.data_dir}/nodes.csv")
            self.edges_df = pd.read_csv(f"{self.data_dir}/edges.csv")
            print(f"数据加载完成：{self.G.number_of_nodes()}个节点，{self.G.number_of_edges()}条边")
            return True
        except FileNotFoundError:
            print("网络文件未找到")
            return False

    def weighted_betweenness_centrality(self, weight_attribute='weight', normalized=True):
        """
        加权介数中心性算法

        Args:
            weight_attribute: 边的权重属性名
            normalized: 是否归一化

        Returns:
            dict: 节点加权介数中心性字典
        """
        print("正在计算加权介数中心性...")

        # 创建边的权重映射
        edge_weights = {}
        for u, v, data in self.G.edges(data=True):
            weight = data.get(weight_attribute, 1.0)
            edge_weights[(u, v)] = weight
            edge_weights[(v, u)] = weight  # 无向图

        # 计算所有节点对之间的最短路径
        betweenness = {node: 0.0 for node in self.G.nodes()}

        nodes_list = list(self.G.nodes())
        n = len(nodes_list)

        for i, source in enumerate(nodes_list):
            if i % 50 == 0:
                print(f"处理节点 {i+1}/{n}...")

            # 使用Dijkstra算法计算从源节点到所有其他节点的最短路径
            distances, paths = nx.single_source_dijkstra(self.G, source, weight=weight_attribute)

            # 计算依赖关系
            dependency = {node: 0.0 for node in self.G.nodes()}

            # 按照距离从大到小处理节点
            sorted_nodes = sorted(distances.keys(), key=lambda x: distances[x], reverse=True)

            for target in sorted_nodes:
                if target == source:
                    continue

                # 找到所有经过source的最短路径
                try:
                    paths_through_source = [path for path in nx.all_shortest_paths(self.G, source, target, weight=weight_attribute)
                                          if source in path and len(path) > 2]
                except:
                    continue

                if not paths_through_source:
                    continue

                # 计算source在最短路径中的贡献
                for path in paths_through_source:
                    if source in path:
                        # 计算路径权重
                        path_weight = 0
                        for j in range(len(path)-1):
                            u, v = path[j], path[j+1]
                            path_weight += edge_weights.get((u, v), 1.0)

                        # 介数中心性贡献（考虑权重）
                        contribution = 1.0 / len(paths_through_source) if paths_through_source else 0
                        dependency[target] += contribution

            # 累加到介数中心性
            for node in dependency:
                if node != source:
                    betweenness[node] += dependency[node]

        # 归一化
        if normalized:
            max_betweenness = max(betweenness.values()) if betweenness else 1
            if max_betweenness > 0:
                betweenness = {node: val / max_betweenness for node, val in betweenness.items()}

        print("加权介数中心性计算完成")
        return betweenness

    def multi_indicator_fusion_model(self, weights=None):
        """
        多指标融合评分模型

        Args:
            weights: 各指标权重字典

        Returns:
            dict: 融合后的关键节点评分
        """
        print("正在构建多指标融合评分模型...")

        if weights is None:
            # 默认权重
            weights = {
                'degree': 0.2,
                'betweenness': 0.25,
                'closeness': 0.15,
                'eigenvector': 0.15,
                'pagerank': 0.15,
                'weighted_betweenness': 0.1
            }

        # 计算各项中心性指标
        print("计算各项中心性指标...")
        centrality_measures = {
            'degree': nx.degree_centrality(self.G),
            'betweenness': nx.betweenness_centrality(self.G),
            'closeness': nx.closeness_centrality(self.G),
            'eigenvector': nx.eigenvector_centrality(self.G, max_iter=1000),
            'pagerank': nx.pagerank(self.G, alpha=0.85)
        }

        # 计算加权介数中心性
        centrality_measures['weighted_betweenness'] = self.weighted_betweenness_centrality()

        # 数据标准化
        scaler = MinMaxScaler()
        centrality_matrix = np.array([list(cent.values()) for cent in centrality_measures.values()]).T
        normalized_matrix = scaler.fit_transform(centrality_matrix)

        # 重建标准化后的中心性字典
        normalized_centralities = {}
        for i, measure_name in enumerate(centrality_measures.keys()):
            normalized_centralities[measure_name] = dict(zip(self.G.nodes(), normalized_matrix[:, i]))

        # 多指标融合评分
        fusion_scores = {}
        for node in self.G.nodes():
            score = 0
            for measure_name, weight in weights.items():
                score += normalized_centralities[measure_name][node] * weight
            fusion_scores[node] = score

        # 再次归一化
        max_score = max(fusion_scores.values()) if fusion_scores else 1
        if max_score > 0:
            fusion_scores = {node: score / max_score for node, score in fusion_scores.items()}

        print("多指标融合评分模型构建完成")
        return fusion_scores, normalized_centralities, weights

    def improved_pagerank_algorithm(self, risk_weights=True, alpha=0.85, max_iter=100):
        """
        改进的PageRank算法（考虑风险等级）

        Args:
            risk_weights: 是否使用风险权重
            alpha: 阻尼系数
            max_iter: 最大迭代次数

        Returns:
            dict: 改进的PageRank评分
        """
        print("正在计算改进的PageRank算法...")

        # 基本PageRank
        base_pagerank = nx.pagerank(self.G, alpha=alpha, max_iter=max_iter)

        if not risk_weights:
            return base_pagerank

        # 考虑风险等级的改进PageRank
        # 风险等级映射
        risk_mapping = {'高': 1.5, '中': 1.0, '低': 0.7}
        risk_scores = {}

        for node in self.G.nodes():
            node_data = self.nodes_df[self.nodes_df['node_id'] == node]
            if not node_data.empty:
                risk_level = node_data['risk_level'].iloc[0]
                risk_scores[node] = risk_mapping.get(risk_level, 1.0)
            else:
                risk_scores[node] = 1.0

        # 个性化向量（基于风险等级）
        personalization = {node: risk_scores[node] for node in self.G.nodes()}

        # 计算改进的PageRank
        improved_pagerank = nx.pagerank(self.G, alpha=alpha, personalization=personalization, max_iter=max_iter)

        return improved_pagerank

    def structural_holes_based_ranking(self):
        """
        基于结构洞的关键节点识别

        Returns:
            dict: 基于结构洞的节点重要性评分
        """
        print("正在计算基于结构洞的关键节点识别...")

        # 计算结构洞指标
        constraint_dict = nx.constraint(self.G)
        effective_size_dict = nx.effective_size(self.G)

        # 结构洞重要性评分（低约束度 + 高有效尺寸）
        structural_scores = {}

        constraints = list(constraint_dict.values())
        effective_sizes = list(effective_size_dict.values())

        # 处理None值
        valid_constraints = [c for c in constraints if c is not None]
        if valid_constraints:
            min_constraint = min(valid_constraints)
            max_constraint = max(valid_constraints)
            constraint_range = max_constraint - min_constraint if max_constraint > min_constraint else 1
        else:
            constraint_range = 1

        if effective_sizes:
            min_eff_size = min(effective_sizes)
            max_eff_size = max(effective_sizes)
            eff_size_range = max_eff_size - min_eff_size if max_eff_size > min_eff_size else 1
        else:
            eff_size_range = 1

        for node in self.G.nodes():
            constraint = constraint_dict.get(node, 1.0)
            eff_size = effective_size_dict.get(node, 0.0)

            # 标准化
            if constraint_range > 0:
                norm_constraint = 1 - (constraint - min_constraint) / constraint_range  # 低约束度得分高
            else:
                norm_constraint = 0.5

            if eff_size_range > 0:
                norm_eff_size = (eff_size - min_eff_size) / eff_size_range
            else:
                norm_eff_size = 0.5

            # 综合评分
            structural_scores[node] = (norm_constraint + norm_eff_size) / 2

        return structural_scores

    def vulnerability_based_ranking(self):
        """
        基于网络脆弱性的关键节点识别

        Returns:
            dict: 基于脆弱性的节点重要性评分
        """
        print("正在计算基于网络脆弱性的关键节点识别...")

        # 计算节点的网络破坏影响
        vulnerability_scores = {}

        # 由于计算量太大，我们使用近似方法
        # 基于度数、介数中心性和特征向量中心性的组合

        degree_cent = nx.degree_centrality(self.G)
        betweenness_cent = nx.betweenness_centrality(self.G)
        eigenvector_cent = nx.eigenvector_centrality(self.G, max_iter=1000)

        for node in self.G.nodes():
            # 脆弱性评分 = 度中心性 * 介数中心性 * 特征向量中心性
            # 这个组合可以较好地反映节点的网络破坏潜力
            vulnerability_scores[node] = (
                degree_cent[node] *
                betweenness_cent[node] *
                eigenvector_cent[node]
            )

        # 归一化
        max_score = max(vulnerability_scores.values()) if vulnerability_scores else 1
        if max_score > 0:
            vulnerability_scores = {node: score / max_score for node, score in vulnerability_scores.items()}

        return vulnerability_scores

    def compare_algorithms(self, ground_truth=None, top_k=50):
        """
        比较不同关键节点识别算法的性能

        Args:
            ground_truth: 真实关键节点标签（如果有的话）
            top_k: 评估前k个节点

        Returns:
            dict: 算法性能比较结果
        """
        print("正在比较关键节点识别算法性能...")

        algorithms = {
            '度中心性': nx.degree_centrality(self.G),
            '介数中心性': nx.betweenness_centrality(self.G),
            '接近中心性': nx.closeness_centrality(self.G),
            '特征向量中心性': nx.eigenvector_centrality(self.G, max_iter=1000),
            'PageRank': nx.pagerank(self.G, alpha=0.85),
            '加权介数中心性': self.weighted_betweenness_centrality(),
            '多指标融合': self.multi_indicator_fusion_model()[0],
            '改进PageRank': self.improved_pagerank_algorithm(),
            '结构洞排名': self.structural_holes_based_ranking(),
            '脆弱性排名': self.vulnerability_based_ranking()
        }

        self.key_node_rankings = algorithms

        # 性能评估
        performance_results = {}

        for alg_name, rankings in algorithms.items():
            # 排序节点（从重要到不重要）
            sorted_nodes = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

            # Top-K节点
            top_k_nodes = [node for node, score in sorted_nodes[:top_k]]

            # 基本统计
            scores = list(rankings.values())
            performance_results[alg_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': max(scores),
                'min_score': min(scores),
                'top_k_nodes': top_k_nodes,
                'score_distribution': scores
            }

            # 如果有真实标签，计算准确性指标
            if ground_truth:
                # 简化的评估：计算Top-K节点中有多少是真实关键节点
                true_key_nodes = set(ground_truth)
                predicted_key_nodes = set(top_k_nodes)
                precision = len(true_key_nodes.intersection(predicted_key_nodes)) / len(predicted_key_nodes) if predicted_key_nodes else 0
                recall = len(true_key_nodes.intersection(predicted_key_nodes)) / len(true_key_nodes) if true_key_nodes else 0

                performance_results[alg_name]['precision'] = precision
                performance_results[alg_name]['recall'] = recall
                performance_results[alg_name]['f1_score'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        self.algorithm_performance = performance_results
        return performance_results

    def create_algorithm_visualizations(self, output_dir="algorithm_analysis"):
        """创建算法分析可视化"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在创建算法分析可视化...")

        # 1. 算法性能对比图
        self.plot_algorithm_comparison(output_dir)

        # 2. Top关键节点对比
        self.plot_top_key_nodes_comparison(output_dir)

        # 3. 算法相关性分析
        self.plot_algorithm_correlations(output_dir)

        # 4. 关键节点重要性分布
        self.plot_key_node_distributions(output_dir)

    def plot_algorithm_comparison(self, output_dir):
        """绘制算法性能对比图"""
        if not self.algorithm_performance:
            return

        plt.figure(figsize=(15, 10))

        # 提取数据
        algorithms = list(self.algorithm_performance.keys())
        mean_scores = [self.algorithm_performance[alg]['mean_score'] for alg in algorithms]
        std_scores = [self.algorithm_performance[alg]['std_score'] for alg in algorithms]
        max_scores = [self.algorithm_performance[alg]['max_score'] for alg in algorithms]

        # 子图1：平均得分对比
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(algorithms)), mean_scores, yerr=std_scores,
                      capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        plt.ylabel('平均重要性得分')
        plt.title('关键节点识别算法平均得分对比')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, mean_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_scores[i] + 0.01,
                    '.3f', ha='center', va='bottom', fontsize=8)

        # 子图2：最大得分对比
        plt.subplot(2, 2, 2)
        bars = plt.bar(range(len(algorithms)), max_scores, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        plt.ylabel('最大重要性得分')
        plt.title('关键节点识别算法最大得分对比')
        plt.grid(True, alpha=0.3)

        # 子图3：得分分布箱线图
        plt.subplot(2, 2, 3)
        data_to_plot = [self.algorithm_performance[alg]['score_distribution'] for alg in algorithms]
        plt.boxplot(data_to_plot, labels=algorithms)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('重要性得分')
        plt.title('关键节点识别算法得分分布')
        plt.grid(True, alpha=0.3)

        # 子图4：算法评估雷达图
        plt.subplot(2, 2, 4)
        # 选择关键指标进行雷达图展示
        categories = ['平均得分', '得分标准差', '最大得分', '得分范围']
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

        for i, alg in enumerate(algorithms[:6]):  # 只显示前6个算法
            perf = self.algorithm_performance[alg]
            values = [
                perf['mean_score'] * 10,  # 放大显示
                perf['std_score'] * 10,
                perf['max_score'] * 10,
                (perf['max_score'] - perf['min_score']) * 10
            ]
            values += values[:1]  # 闭合雷达图
            angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories) + 1)]

            if i == 0:
                ax.plot(angles, values, 'o-', linewidth=2, label=alg, markersize=6)
            else:
                ax.plot(angles, values, 'o-', linewidth=1, label=alg, markersize=4, alpha=0.7)

        ax.set_thetagrids([a * 180/np.pi for a in angles[:-1]], categories)
        ax.set_title('算法性能雷达图', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/algorithm_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_top_key_nodes_comparison(self, output_dir):
        """绘制Top关键节点对比图"""
        if not self.key_node_rankings:
            return

        plt.figure(figsize=(16, 12))

        # 选择前8个算法
        selected_algorithms = list(self.key_node_rankings.keys())[:8]
        n_algorithms = len(selected_algorithms)

        # 为每个算法显示Top 10节点
        for i, alg_name in enumerate(selected_algorithms):
            plt.subplot(4, 2, i+1)

            rankings = self.key_node_rankings[alg_name]
            sorted_nodes = sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:10]

            nodes = [f'S{node[1:]}' for node, score in sorted_nodes]
            scores = [score for node, score in sorted_nodes]

            bars = plt.barh(range(len(nodes)), scores, alpha=0.7, color='steelblue', edgecolor='black')
            plt.yticks(range(len(nodes)), nodes)
            plt.xlabel('重要性得分')
            plt.title(f'{alg_name} - Top 10关键节点')
            plt.grid(True, alpha=0.3)

            # 添加数值标签
            for j, (bar, score) in enumerate(zip(bars, scores)):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        '.3f', ha='left', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_key_nodes_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_algorithm_correlations(self, output_dir):
        """绘制算法相关性分析图"""
        if not self.key_node_rankings:
            return

        plt.figure(figsize=(12, 10))

        # 计算算法之间的相关性
        algorithms = list(self.key_node_rankings.keys())
        n_algorithms = len(algorithms)

        correlation_matrix = np.zeros((n_algorithms, n_algorithms))

        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                scores1 = list(self.key_node_rankings[alg1].values())
                scores2 = list(self.key_node_rankings[alg2].values())
                corr = np.corrcoef(scores1, scores2)[0, 1]
                correlation_matrix[i, j] = corr

        # 绘制热力图
        plt.subplot(1, 1, 1)
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=algorithms, yticklabels=algorithms,
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('关键节点识别算法相关性矩阵')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/algorithm_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_key_node_distributions(self, output_dir):
        """绘制关键节点重要性分布图"""
        if not self.key_node_rankings:
            return

        plt.figure(figsize=(15, 12))

        # 选择6个代表性算法
        selected_algorithms = ['度中心性', '介数中心性', '特征向量中心性',
                              '加权介数中心性', '多指标融合', '改进PageRank']

        for i, alg_name in enumerate(selected_algorithms):
            if alg_name not in self.key_node_rankings:
                continue

            plt.subplot(3, 2, i+1)

            scores = list(self.key_node_rankings[alg_name].values())

            # 分布直方图
            plt.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('重要性得分')
            plt.ylabel('频次')
            plt.title(f'{alg_name}重要性分布')
            plt.grid(True, alpha=0.3)

            # 添加统计信息
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'均值: {mean_val:.3f}')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/key_node_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_algorithm_report(self, output_dir="algorithm_analysis"):
        """生成算法分析报告"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在生成关键节点识别算法报告...")

        with open(f"{output_dir}/key_node_algorithm_report.txt", 'w', encoding='utf-8') as f:
            f.write("=== 电信诈骗犯罪网络关键节点识别算法分析报告 ===\n\n")

            if self.algorithm_performance:
                f.write("1. 算法性能对比:\n")
                for alg_name, perf in self.algorithm_performance.items():
                    f.write(f"   {alg_name}:\n")
                f.write(f"      平均得分: {perf['mean_score']:.6f}\n")
                f.write(f"      得分标准差: {perf['std_score']:.6f}\n")
                f.write(f"      最大得分: {perf['max_score']:.6f}\n")
                f.write(f"      最小得分: {perf['min_score']:.6f}\n")
                if 'precision' in perf:
                    f.write(f"      精确率: {perf['precision']:.4f}\n")
                    f.write(f"      召回率: {perf['recall']:.4f}\n")
                    f.write(f"      F1得分: {perf['f1_score']:.4f}\n")
                f.write(f"      Top 10关键节点: {[f'S{node[1:]}' for node in perf['top_k_nodes'][:10]]}\n")
                f.write("\n")

            f.write("2. 算法评价与建议:\n")

            # 基于性能数据的评价
            if self.algorithm_performance:
                # 找出表现最好的算法
                best_algorithm = max(self.algorithm_performance.items(),
                                   key=lambda x: x[1]['mean_score'])

                f.write(f"   • 综合表现最佳算法: {best_algorithm[0]} (平均得分: {best_algorithm[1]['mean_score']:.4f})\n")

                # 分析不同算法的特点
                f.write("   • 算法特点分析:\n")
                f.write("     - 度中心性: 简单直接，计算效率高，适合发现局部重要节点\n")
                f.write("     - 介数中心性: 能发现网络中的'桥梁'节点，对网络连通性影响大\n")
                f.write("     - 接近中心性: 反映节点到其他节点的平均距离，适合发现全局中心节点\n")
                f.write("     - 特征向量中心性: 考虑邻居的重要性，适合发现影响力大的节点\n")
                f.write("     - PageRank: 类似于网络搜索中的重要性排序\n")
                f.write("     - 加权介数中心性: 考虑边的权重，更适合现实网络\n")
                f.write("     - 多指标融合: 综合多种指标，更全面地评估节点重要性\n")
                f.write("     - 改进PageRank: 结合风险等级信息，更适合犯罪网络分析\n")

            f.write("\n3. 关键节点识别策略建议:\n")
            f.write("   • 优先级排序: 多指标融合 > 介数中心性 > 特征向量中心性 > 度中心性\n")
            f.write("   • 应用场景: \n")
            f.write("     - 快速打击: 使用度中心性和介数中心性\n")
            f.write("     - 深度分析: 使用多指标融合和改进PageRank\n")
            f.write("     - 预防重点: 重点监控高风险等级的中心节点\n")
            f.write("   • 组合应用: 将多种算法结果结合使用，避免单一算法的局限性\n")

            f.write("\n4. 算法优化方向:\n")
            f.write("   • 融入领域知识: 结合犯罪类型、地理位置等特征\n")
            f.write("   • 时间动态性: 考虑网络随时间的变化\n")
            f.write("   • 多网络融合: 结合通信网络、资金网络等不同类型的网络\n")
            f.write("   • 机器学习方法: 使用图神经网络等先进技术\n")

        print(f"关键节点识别算法报告已保存到 {output_dir}/key_node_algorithm_report.txt")

    def run_complete_analysis(self):
        """运行完整关键节点识别分析"""
        print("=== 开始电信诈骗犯罪网络关键节点识别算法分析 ===\n")

        if not self.load_data():
            return False

        # 比较不同算法
        self.compare_algorithms()

        # 创建可视化
        self.create_algorithm_visualizations()

        # 生成分析报告
        self.generate_algorithm_report()

        print("\n=== 关键节点识别算法分析完成 ===")
        print("生成的文件:")
        print("- algorithm_analysis/ 目录：算法分析可视化图表")
        print("- algorithm_analysis/key_node_algorithm_report.txt：算法分析报告")

        return True


def main():
    """主函数"""
    identifier = KeyNodeIdentifier()
    identifier.run_complete_analysis()


if __name__ == "__main__":
    main()
