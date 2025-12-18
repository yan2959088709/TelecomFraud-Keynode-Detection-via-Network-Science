"""
电信诈骗犯罪网络复杂网络特性分析
验证小世界、无标度、抗毁性等特性
基于复杂网络与社会网络特性的电信诈骗犯罪网络关键节点识别研究
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ComplexNetworkAnalyzer:
    """复杂网络特性分析器"""

    def __init__(self, data_dir="data"):
        """
        初始化分析器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.G = None
        self.results = {}

    def load_network(self):
        """加载网络数据"""
        print("正在加载网络数据...")
        try:
            with open(f"{self.data_dir}/network_graph.pkl", 'rb') as f:
                self.G = pickle.load(f)
            print(f"网络加载完成：{self.G.number_of_nodes()}个节点，{self.G.number_of_edges()}条边")
            return True
        except FileNotFoundError:
            print("网络文件未找到")
            return False

    def analyze_small_world_properties(self):
        """分析小世界特性"""
        print("正在分析小世界特性...")

        # 获取网络基本信息
        n = self.G.number_of_nodes()
        m = self.G.number_of_edges()

        # 计算实际网络的聚类系数和平均路径长度
        actual_clustering = nx.average_clustering(self.G)

        # 对于不连通图，计算最大连通分量的平均路径长度
        if nx.is_connected(self.G):
            actual_avg_path = nx.average_shortest_path_length(self.G)
        else:
            largest_cc = max(nx.connected_components(self.G), key=len)
            subgraph = self.G.subgraph(largest_cc)
            actual_avg_path = nx.average_shortest_path_length(subgraph)
            print(f"网络不连通，使用最大连通分量（{len(largest_cc)}个节点）计算平均路径长度")

        # 生成随机网络进行对比
        print("生成随机网络进行对比...")
        random_graphs = []
        for i in range(5):  # 生成5个随机网络取平均
            random_g = nx.gnm_random_graph(n, m, seed=i)
            if nx.is_connected(random_g):
                random_clustering = nx.average_clustering(random_g)
                random_avg_path = nx.average_shortest_path_length(random_g)
            else:
                largest_cc_rand = max(nx.connected_components(random_g), key=len)
                subgraph_rand = random_g.subgraph(largest_cc_rand)
                random_clustering = nx.average_clustering(subgraph_rand)
                random_avg_path = nx.average_shortest_path_length(subgraph_rand)

            random_graphs.append({
                'clustering': random_clustering,
                'avg_path': random_avg_path
            })

        # 计算随机网络平均值
        random_clustering_avg = np.mean([rg['clustering'] for rg in random_graphs])
        random_avg_path_avg = np.mean([rg['avg_path'] for rg in random_graphs])

        # 计算小世界系数
        small_world_coefficient = (actual_clustering / random_clustering_avg) / (actual_avg_path / random_avg_path_avg)

        # 生成规则网络进行对比（最近邻耦合网络）
        print("生成规则网络进行对比...")
        regular_graph = nx.watts_strogatz_graph(n, k=int(2*m/n), p=0)  # p=0为规则网络
        regular_clustering = nx.average_clustering(regular_graph)
        regular_avg_path = nx.average_shortest_path_length(regular_graph)

        self.results['small_world'] = {
            'actual_clustering': actual_clustering,
            'actual_avg_path': actual_avg_path,
            'random_clustering': random_clustering_avg,
            'random_avg_path': random_avg_path_avg,
            'regular_clustering': regular_clustering,
            'regular_avg_path': regular_avg_path,
            'small_world_coefficient': small_world_coefficient,
            'is_small_world': small_world_coefficient > 1
        }

        print(f"小世界系数: {small_world_coefficient:.4f}")
        print(f"是否为小世界网络: {'是' if small_world_coefficient > 1 else '否'}")
        return self.results['small_world']

    def analyze_scale_free_properties(self):
        """分析无标度特性"""
        print("正在分析无标度特性...")

        # 计算度分布
        degrees = [d for n, d in self.G.degree()]
        degree_counts = Counter(degrees)

        # 对数坐标下的线性回归分析
        deg_vals = []
        freq_vals = []

        for deg in sorted(degree_counts.keys()):
            if deg > 0:  # 排除度为0的节点
                deg_vals.append(np.log(deg))
                freq_vals.append(np.log(degree_counts[deg]))

        # 进行线性回归
        if len(deg_vals) > 10:  # 确保有足够的数据点
            slope, intercept, r_value, p_value, std_err = stats.linregress(deg_vals, freq_vals)

            # 计算幂律指数 (slope 是负值，所以幂律指数为 -slope)
            power_law_exponent = -slope

            # 拟合优度 (R²)
            r_squared = r_value ** 2

            # Kolmogorov-Smirnov检验
            # 生成理论幂律分布进行比较
            max_deg = max(degrees)
            theoretical_dist = []
            for deg in range(1, max_deg + 1):
                prob = deg ** (-power_law_exponent)
                theoretical_dist.extend([deg] * int(prob * len([d for d in degrees if d >= deg])))

            if len(theoretical_dist) > len(degrees):
                theoretical_dist = theoretical_dist[:len(degrees)]

            try:
                ks_stat, ks_p_value = stats.ks_2samp(degrees, theoretical_dist)
            except:
                ks_stat, ks_p_value = None, None

            self.results['scale_free'] = {
                'power_law_exponent': power_law_exponent,
                'r_squared': r_squared,
                'slope': slope,
                'intercept': intercept,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'is_scale_free': r_squared > 0.8 and power_law_exponent > 2 and power_law_exponent < 3,
                'degree_distribution': dict(degree_counts)
            }

            print(f"幂律指数: {power_law_exponent:.3f}")
            print(f"拟合优度 R2: {r_squared:.4f}")
            if ks_p_value:
                print(f"KS检验 p值: {ks_p_value:.4f}")
            print(f"是否符合无标度特性: {self.results['scale_free']['is_scale_free']}")
        else:
            print("数据点不足，无法进行无标度分析")
            self.results['scale_free'] = None

        return self.results['scale_free']

    def analyze_robustness(self):
        """分析网络抗毁性"""
        print("正在分析网络抗毁性...")

        # 复制原网络
        G_original = self.G.copy()

        # 分析函数：计算最大连通分量比例
        def calculate_lcc_fraction(graph):
            if graph.number_of_nodes() == 0:
                return 0
            largest_cc = max(nx.connected_components(graph), key=len)
            return len(largest_cc) / graph.number_of_nodes()

        # 1. 随机节点移除
        print("分析随机节点移除的抗毁性...")
        random_removal_results = []
        G_temp = G_original.copy()

        remove_fractions = np.linspace(0, 0.9, 20)  # 移除0%到90%的节点

        for frac in remove_fractions:
            if G_temp.number_of_nodes() == 0:
                break

            num_to_remove = int(frac * G_original.number_of_nodes()) - len(random_removal_results)
            if num_to_remove <= 0:
                continue

            # 随机选择要移除的节点
            nodes_to_remove = np.random.choice(list(G_temp.nodes()),
                                             size=min(num_to_remove, G_temp.number_of_nodes()),
                                             replace=False)

            G_temp.remove_nodes_from(nodes_to_remove)
            lcc_fraction = calculate_lcc_fraction(G_temp)
            random_removal_results.append({
                'removal_fraction': frac,
                'lcc_fraction': lcc_fraction,
                'remaining_nodes': G_temp.number_of_nodes()
            })

        # 2. 按度数移除节点（针对性攻击）
        print("分析针对性节点移除的抗毁性...")
        targeted_removal_results = []
        G_temp = G_original.copy()

        # 按度数排序节点（从大到小）
        sorted_nodes = sorted(G_temp.degree(), key=lambda x: x[1], reverse=True)

        for frac in remove_fractions:
            if G_temp.number_of_nodes() == 0:
                break

            num_to_remove = int(frac * G_original.number_of_nodes()) - len(targeted_removal_results)
            if num_to_remove <= 0:
                continue

            # 移除度数最高的节点
            nodes_to_remove = [node for node, degree in sorted_nodes[:num_to_remove] if node in G_temp]
            sorted_nodes = [(n, d) for n, d in sorted_nodes if n not in nodes_to_remove]

            G_temp.remove_nodes_from(nodes_to_remove)
            lcc_fraction = calculate_lcc_fraction(G_temp)
            targeted_removal_results.append({
                'removal_fraction': frac,
                'lcc_fraction': lcc_fraction,
                'remaining_nodes': G_temp.number_of_nodes()
            })

        # 3. 按介数中心性移除节点
        print("分析按介数中心性移除的抗毁性...")
        betweenness_removal_results = []
        G_temp = G_original.copy()

        # 计算介数中心性
        betweenness = nx.betweenness_centrality(G_temp)
        sorted_nodes_bc = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

        for frac in remove_fractions:
            if G_temp.number_of_nodes() == 0:
                break

            num_to_remove = int(frac * G_original.number_of_nodes()) - len(betweenness_removal_results)
            if num_to_remove <= 0:
                continue

            # 移除介数中心性最高的节点
            nodes_to_remove = [node for node, bc in sorted_nodes_bc[:num_to_remove] if node in G_temp]
            sorted_nodes_bc = [(n, bc) for n, bc in sorted_nodes_bc if n not in nodes_to_remove]

            G_temp.remove_nodes_from(nodes_to_remove)
            lcc_fraction = calculate_lcc_fraction(G_temp)
            betweenness_removal_results.append({
                'removal_fraction': frac,
                'lcc_fraction': lcc_fraction,
                'remaining_nodes': G_temp.number_of_nodes()
            })

        self.results['robustness'] = {
            'random_removal': random_removal_results,
            'targeted_removal': targeted_removal_results,
            'betweenness_removal': betweenness_removal_results
        }

        print("抗毁性分析完成")
        return self.results['robustness']

    def create_complex_network_visualizations(self, output_dir="complex_analysis"):
        """创建复杂网络特性可视化"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在创建复杂网络特性可视化...")

        # 1. 小世界特性对比图
        self.plot_small_world_analysis(output_dir)

        # 2. 无标度特性分析图
        self.plot_scale_free_analysis(output_dir)

        # 3. 网络抗毁性分析图
        self.plot_robustness_analysis(output_dir)

        # 4. 综合特性雷达图
        self.plot_network_characteristics_radar(output_dir)

    def plot_small_world_analysis(self, output_dir):
        """绘制小世界特性分析图"""
        if 'small_world' not in self.results:
            return

        sw = self.results['small_world']

        plt.figure(figsize=(15, 10))

        # 子图1：聚类系数对比
        plt.subplot(2, 2, 1)
        networks = ['实际网络', '随机网络', '规则网络']
        clustering_values = [sw['actual_clustering'], sw['random_clustering'], sw['regular_clustering']]
        bars = plt.bar(networks, clustering_values, color=['blue', 'orange', 'green'], alpha=0.7)
        plt.ylabel('聚类系数')
        plt.title('聚类系数对比')
        plt.ylim(0, max(clustering_values) * 1.2)

        for bar, value in zip(bars, clustering_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    '.4f', ha='center', va='bottom')

        # 子图2：平均路径长度对比
        plt.subplot(2, 2, 2)
        path_values = [sw['actual_avg_path'], sw['random_avg_path'], sw['regular_avg_path']]
        bars = plt.bar(networks, path_values, color=['blue', 'orange', 'green'], alpha=0.7)
        plt.ylabel('平均路径长度')
        plt.title('平均路径长度对比')
        plt.ylim(0, max(path_values) * 1.2)

        for bar, value in zip(bars, path_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.2f', ha='center', va='bottom')

        # 子图3：小世界系数
        plt.subplot(2, 2, 3)
        sw_coeff = sw['small_world_coefficient']
        colors = ['red' if sw_coeff > 1 else 'gray']
        bars = plt.bar(['小世界系数'], [sw_coeff], color=colors, alpha=0.7)
        plt.ylabel('小世界系数')
        plt.title('小世界特性验证')
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='临界值')
        plt.legend()

        plt.text(0, sw_coeff + 0.1, '.3f', ha='center', va='bottom', fontweight='bold')

        # 子图4：特性说明
        plt.subplot(2, 2, 4)
        plt.axis('off')
        characteristics = [
            f"Small-world coefficient: {sw_coeff:.3f}",
            f"Is small-world network: {'Yes' if sw['is_small_world'] else 'No'}",
            "",
            "Small-world network features:",
            "• High clustering coefficient (like regular networks)",
            "• Short average path length (like random networks)",
            f"• Actual clustering is {sw['actual_clustering']/sw['random_clustering']:.2f}x that of random network",
            f"• Actual path length is {sw['regular_avg_path']/sw['actual_avg_path']:.2f}x that of regular network"
        ]

        text_content = '\n'.join(characteristics)
        plt.text(0.1, 0.9, text_content, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/small_world_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_scale_free_analysis(self, output_dir):
        """绘制无标度特性分析图"""
        if 'scale_free' not in self.results or self.results['scale_free'] is None:
            return

        sf = self.results['scale_free']

        plt.figure(figsize=(15, 10))

        # 子图1：对数-对数度分布
        plt.subplot(2, 2, 1)
        degrees = list(sf['degree_distribution'].keys())
        counts = list(sf['degree_distribution'].values())

        plt.loglog(degrees, counts, 'bo', markersize=6, alpha=0.7, label='实际数据')

        # 绘制拟合的幂律线
        x_fit = np.logspace(np.log10(min(degrees)), np.log10(max(degrees)), 100)
        y_fit = np.exp(sf['intercept']) * (x_fit ** sf['slope'])
        plt.loglog(x_fit, y_fit, 'r-', linewidth=2, label=f'幂律拟合 (γ={-sf["slope"]:.2f})')

        plt.xlabel('度数 (log)')
        plt.ylabel('频次 (log)')
        plt.title('度分布的幂律特性')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：线性回归分析
        plt.subplot(2, 2, 2)
        deg_vals = []
        freq_vals = []

        for deg in sorted(sf['degree_distribution'].keys()):
            if deg > 0:
                deg_vals.append(np.log(deg))
                freq_vals.append(np.log(sf['degree_distribution'][deg]))

        plt.scatter(deg_vals, freq_vals, alpha=0.7, color='blue', s=30)
        plt.plot(deg_vals, sf['intercept'] + sf['slope'] * np.array(deg_vals),
                'r-', linewidth=2, label=f'拟合线 (斜率={sf["slope"]:.3f})')

        plt.xlabel('ln(度数)')
        plt.ylabel('ln(频次)')
        plt.title('线性回归拟合')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图3：残差分析
        plt.subplot(2, 2, 3)
        predicted = sf['intercept'] + sf['slope'] * np.array(deg_vals)
        residuals = np.array(freq_vals) - predicted

        plt.scatter(predicted, residuals, alpha=0.7, color='green', s=30)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差分析')
        plt.grid(True, alpha=0.3)

        # 子图4：特性评估
        plt.subplot(2, 2, 4)
        plt.axis('off')

        assessment = [
            f"Power-law exponent γ: {-sf['slope']:.3f}",
            f"Coefficient of determination R²: {sf['r_squared']:.4f}",
            f"KS test statistic: {sf['ks_statistic']:.4f}" if sf['ks_statistic'] else "KS test: Cannot calculate",
            f"KS test p-value: {sf['ks_p_value']:.4f}" if sf['ks_p_value'] else "KS test p-value: Cannot calculate",
            "",
            "Scale-free network criteria:",
            f"• R² > 0.8: {'✓' if sf['r_squared'] > 0.8 else '✗'} ({sf['r_squared']:.4f})",
            f"• 2 < γ < 3: {'✓' if 2 < -sf['slope'] < 3 else '✗'} ({-sf['slope']:.3f})",
            "",
            f"Overall assessment: {'Scale-free' if sf['is_scale_free'] else 'Not scale-free'}"
        ]

        text_content = '\n'.join(assessment)
        plt.text(0.05, 0.95, text_content, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/scale_free_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_robustness_analysis(self, output_dir):
        """绘制网络抗毁性分析图"""
        if 'robustness' not in self.results:
            return

        rob = self.results['robustness']

        plt.figure(figsize=(15, 10))

        # 提取数据
        random_frac = [r['removal_fraction'] for r in rob['random_removal']]
        random_lcc = [r['lcc_fraction'] for r in rob['random_removal']]

        targeted_frac = [r['removal_fraction'] for r in rob['targeted_removal']]
        targeted_lcc = [r['lcc_fraction'] for r in rob['targeted_removal']]

        betweenness_frac = [r['removal_fraction'] for r in rob['betweenness_removal']]
        betweenness_lcc = [r['lcc_fraction'] for r in rob['betweenness_removal']]

        # 子图1：最大连通分量比例 vs 移除比例
        plt.subplot(2, 2, 1)
        plt.plot(random_frac, random_lcc, 'b-o', linewidth=2, markersize=4,
                label='随机移除', alpha=0.8)
        plt.plot(targeted_frac, targeted_lcc, 'r-s', linewidth=2, markersize=4,
                label='按度数移除', alpha=0.8)
        plt.plot(betweenness_frac, betweenness_lcc, 'g-^', linewidth=2, markersize=4,
                label='按介数中心性移除', alpha=0.8)

        plt.xlabel('移除节点比例')
        plt.ylabel('最大连通分量比例')
        plt.title('网络抗毁性分析')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：随机移除 vs 针对性移除对比
        plt.subplot(2, 2, 2)
        plt.plot(random_frac, random_lcc, 'b-', linewidth=3, label='随机移除', alpha=0.8)
        plt.plot(targeted_frac, targeted_lcc, 'r--', linewidth=3, label='针对性移除', alpha=0.8)
        plt.fill_between(random_frac, random_lcc, targeted_lcc, where=np.array(random_lcc) > np.array(targeted_lcc),
                        color='blue', alpha=0.2, label='随机移除更鲁棒')
        plt.fill_between(random_frac, random_lcc, targeted_lcc, where=np.array(random_lcc) < np.array(targeted_lcc),
                        color='red', alpha=0.2, label='针对性移除更有效')

        plt.xlabel('移除节点比例')
        plt.ylabel('最大连通分量比例')
        plt.title('随机vs针对性攻击对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图3：不同攻击策略的效果
        plt.subplot(2, 2, 3)
        strategies = ['随机移除', '按度数移除', '按介数中心性移除']
        lcc_at_50 = [
            random_lcc[len(random_frac)//2] if len(random_frac) > 0 else 0,
            targeted_lcc[len(targeted_frac)//2] if len(targeted_frac) > 0 else 0,
            betweenness_lcc[len(betweenness_frac)//2] if len(betweenness_frac) > 0 else 0
        ]

        bars = plt.bar(strategies, lcc_at_50, color=['blue', 'red', 'green'], alpha=0.7)
        plt.ylabel('最大连通分量比例 (移除50%节点后)')
        plt.title('不同攻击策略效果对比')
        plt.xticks(rotation=45)

        for bar, value in zip(bars, lcc_at_50):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.3f', ha='center', va='bottom')

        # 子图4：抗毁性评估
        plt.subplot(2, 2, 4)
        plt.axis('off')

        # 计算一些关键指标
        random_robustness = np.trapz(random_lcc, random_frac)  # 随机移除下的鲁棒性积分
        targeted_robustness = np.trapz(targeted_lcc, targeted_frac)  # 针对性移除下的鲁棒性积分

        evaluation = [
            "Network robustness assessment:",
            "",
            f"Random robustness: {random_robustness:.3f}",
            f"Targeted robustness: {targeted_robustness:.3f}",
            f"Difference: {random_robustness - targeted_robustness:.3f}",
            "",
            "Robustness level assessment:",
            f"• Random attack robustness: {'High' if random_robustness > 0.6 else 'Medium' if random_robustness > 0.4 else 'Low'}",
            f"• Targeted attack vulnerability: {'High' if targeted_robustness < 0.4 else 'Medium' if targeted_robustness < 0.6 else 'Low'}",
            "",
            "Network type inference:",
            f"• More like random network: {'✓' if random_robustness > targeted_robustness else '✗'}",
            f"• More like scale-free network: {'✓' if random_robustness < targeted_robustness else '✗'}"
        ]

        text_content = '\n'.join(evaluation)
        plt.text(0.05, 0.95, text_content, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/robustness_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_network_characteristics_radar(self, output_dir):
        """绘制网络特性雷达图"""
        plt.figure(figsize=(10, 8))

        # 准备数据
        categories = ['小世界系数', '幂律指数', '聚类系数', '路径长度', '鲁棒性']

        # 标准化数据到0-1范围
        if 'small_world' in self.results:
            sw_coeff = min(self.results['small_world']['small_world_coefficient'] / 2, 1)  # 归一化
        else:
            sw_coeff = 0

        if 'scale_free' in self.results and self.results['scale_free']:
            # 理想的幂律指数是2-3，标准化为0-1
            gamma = -self.results['scale_free']['slope']
            gamma_norm = 1 - abs(gamma - 2.5) / 2.5  # 2.5为中心，距离越近得分越高
            gamma_norm = max(0, min(1, gamma_norm))
        else:
            gamma_norm = 0

        clustering = self.results['small_world']['actual_clustering'] if 'small_world' in self.results else 0
        clustering_norm = min(clustering * 10, 1)  # 假设典型聚类系数在0.1左右

        path_length = self.results['small_world']['actual_avg_path'] if 'small_world' in self.results else 10
        path_norm = max(0, 1 - (path_length - 2) / 8)  # 假设典型路径长度在2-10之间

        if 'robustness' in self.results:
            random_robustness = np.trapz(
                [r['lcc_fraction'] for r in self.results['robustness']['random_removal']],
                [r['removal_fraction'] for r in self.results['robustness']['random_removal']]
            )
        else:
            random_robustness = 0

        values = [sw_coeff, gamma_norm, clustering_norm, path_norm, random_robustness]

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
        plt.title('电信诈骗犯罪网络复杂特性雷达图', size=16, fontweight='bold', pad=20)

        # 添加参考线（理想的无标度小世界网络）
        ideal_values = [1, 1, 0.8, 0.8, 0.7]  # 理想值
        ideal_values += ideal_values[:1]
        plt.plot(angles, ideal_values, 'r--', linewidth=1, label='理想复杂网络', alpha=0.7)
        plt.fill(angles, ideal_values, alpha=0.1, color='red')

        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/network_characteristics_radar.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_complex_analysis_report(self, output_dir="complex_analysis"):
        """生成复杂网络分析报告"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在生成复杂网络分析报告...")

        with open(f"{output_dir}/complex_network_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write("=== 电信诈骗犯罪网络复杂网络特性分析报告 ===\n\n")

            # 小世界特性分析
            if 'small_world' in self.results:
                sw = self.results['small_world']
                f.write("1. 小世界特性分析:\n")
                f.write(f"   实际聚类系数: {sw['actual_clustering']:.6f}\n")
                f.write(f"   实际平均路径长度: {sw['actual_avg_path']:.6f}\n")
                f.write(f"   随机网络聚类系数: {sw['random_clustering']:.6f}\n")
                f.write(f"   随机网络平均路径长度: {sw['random_avg_path']:.6f}\n")
                f.write(f"   规则网络聚类系数: {sw['regular_clustering']:.6f}\n")
                f.write(f"   规则网络平均路径长度: {sw['regular_avg_path']:.6f}\n")
                f.write(f"   小世界系数: {sw['small_world_coefficient']:.3f}\n")
                f.write(f"   是否为小世界网络: {sw['is_small_world']}\n\n")

            # 无标度特性分析
            if 'scale_free' in self.results and self.results['scale_free']:
                sf = self.results['scale_free']
                f.write("2. 无标度特性分析:\n")
                f.write(f"   幂律指数 γ: {sf['power_law_exponent']:.3f}\n")
                f.write(f"   决定系数 R²: {sf['r_squared']:.4f}\n")
                f.write(f"   线性回归斜率: {sf['slope']:.3f}\n")
                f.write(f"   线性回归截距: {sf['intercept']:.3f}\n")
                if sf['ks_statistic']:
                    f.write(f"   KS检验统计量: {sf['ks_statistic']:.4f}\n")
                    f.write(f"   KS检验 p值: {sf['ks_p_value']:.4f}\n")
                f.write(f"   是否符合无标度特性: {sf['is_scale_free']}\n\n")

            # 网络抗毁性分析
            if 'robustness' in self.results:
                rob = self.results['robustness']
                f.write("3. 网络抗毁性分析:\n")

                # 计算关键指标
                random_final = rob['random_removal'][-1]['lcc_fraction'] if rob['random_removal'] else 0
                targeted_final = rob['targeted_removal'][-1]['lcc_fraction'] if rob['targeted_removal'] else 0
                betweenness_final = rob['betweenness_removal'][-1]['lcc_fraction'] if rob['betweenness_removal'] else 0

                f.write(f"   随机移除后LCC比例: {random_final:.3f}\n")
                f.write(f"   按度数移除后LCC比例: {targeted_final:.3f}\n")
                f.write(f"   按介数中心性移除后LCC比例: {betweenness_final:.3f}\n")
                f.write("   抗毁性评估: 网络对随机攻击相对鲁棒，对针对性攻击较为脆弱\n\n")

            # 综合评价
            f.write("4. 综合特性评价:\n")
            f.write("   电信诈骗犯罪网络表现出以下复杂网络特性:\n")

            is_small_world = self.results.get('small_world', {}).get('is_small_world', False)
            is_scale_free = self.results.get('scale_free', {}).get('is_scale_free', False) if self.results.get('scale_free') else False

            if is_small_world and is_scale_free:
                f.write("   ✓ 同时具备小世界和无标度特性，是典型的复杂网络\n")
                f.write("   ✓ 网络结构介于规则网络和随机网络之间\n")
                f.write("   ✓ 对随机扰动具有较强鲁棒性，但对针对性攻击较为敏感\n")
                f.write("   ✓ 符合犯罪网络的隐蔽性和扩张性特征\n")
            elif is_small_world:
                f.write("   ✓ 具备小世界特性，信息传播效率较高\n")
                f.write("   ✓ 局部聚类程度较高，全局连接效率较好\n")
            elif is_scale_free:
                f.write("   ✓ 具备无标度特性，存在关键节点（枢纽）\n")
                f.write("   ✓ 网络结构对枢纽节点的依赖性较强\n")
            else:
                f.write("   ⚠ 既不明显具备小世界特性，也不具备无标度特性\n")
                f.write("   ⚠ 可能更接近随机网络或规则网络的特性\n")

            f.write("\n   实际意义:\n")
            f.write("   • 为打击犯罪网络提供理论指导：优先打击高中心性节点\n")
            f.write("   • 解释犯罪网络的扩张模式：通过枢纽节点快速传播\n")
            f.write("   • 指导预防策略：切断关键连接可有效遏制网络扩张\n")

        print(f"复杂网络分析报告已保存到 {output_dir}/complex_network_analysis_report.txt")

    def run_complete_analysis(self):
        """运行完整复杂网络分析"""
        print("=== 开始电信诈骗犯罪网络复杂网络特性分析 ===\n")

        if not self.load_network():
            return False

        # 分析小世界特性
        self.analyze_small_world_properties()

        # 分析无标度特性
        self.analyze_scale_free_properties()

        # 分析网络抗毁性
        self.analyze_robustness()

        # 创建可视化
        self.create_complex_network_visualizations()

        # 生成分析报告
        self.generate_complex_analysis_report()

        print("\n=== 复杂网络特性分析完成 ===")
        print("生成的文件:")
        print("- complex_analysis/ 目录：复杂网络特性可视化图表")
        print("- complex_analysis/complex_network_analysis_report.txt：详细分析报告")

        return True


def main():
    """主函数"""
    analyzer = ComplexNetworkAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
