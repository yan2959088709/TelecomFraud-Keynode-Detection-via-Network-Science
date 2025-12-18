"""
电信诈骗犯罪网络描述与可视化分析
基于复杂网络与社会网络特性的电信诈骗犯罪网络关键节点识别研究
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TelecomFraudNetworkAnalyzer:
    """电信诈骗网络分析器"""

    def __init__(self, data_dir="data"):
        """
        初始化分析器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.nodes_df = None
        self.edges_df = None
        self.G = None

        # 分析结果存储
        self.basic_metrics = {}
        self.social_metrics = {}
        self.node_centralities = {}

    def load_data(self):
        """加载网络数据"""
        print("正在加载网络数据...")

        try:
            self.nodes_df = pd.read_csv(f"{self.data_dir}/nodes.csv")
            self.edges_df = pd.read_csv(f"{self.data_dir}/edges.csv")

            # 加载NetworkX图对象
            with open(f"{self.data_dir}/network_graph.pkl", 'rb') as f:
                self.G = pickle.load(f)

            print(f"数据加载完成：{len(self.nodes_df)}个节点，{len(self.edges_df)}条边")

        except FileNotFoundError as e:
            print(f"数据文件未找到：{e}")
            return False

        return True

    def calculate_basic_metrics(self):
        """计算网络基础指标"""
        print("正在计算网络基础指标...")

        self.basic_metrics = {
            # 基本拓扑指标
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'average_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),

            # 连通性指标
            'is_connected': nx.is_connected(self.G),
            'num_components': nx.number_connected_components(self.G),

            # 路径相关指标
            'average_shortest_path_length': nx.average_shortest_path_length(self.G) if nx.is_connected(self.G) else None,
            'diameter': nx.diameter(self.G) if nx.is_connected(self.G) else None,

            # 聚类指标
            'average_clustering': nx.average_clustering(self.G),
            'transitivity': nx.transitivity(self.G),

            # 度分布统计
            'degree_assortativity': nx.degree_assortativity_coefficient(self.G)
        }

        # 计算度分布
        degrees = [d for n, d in self.G.degree()]
        self.basic_metrics['degree_stats'] = {
            'min_degree': min(degrees),
            'max_degree': max(degrees),
            'mean_degree': np.mean(degrees),
            'median_degree': np.median(degrees),
            'std_degree': np.std(degrees)
        }

        return self.basic_metrics

    def calculate_social_network_metrics(self):
        """计算社会网络指标"""
        print("正在计算社会网络指标...")

        # 各种中心性度量
        self.social_metrics = {
            'degree_centrality': nx.degree_centrality(self.G),
            'betweenness_centrality': nx.betweenness_centrality(self.G),
            'closeness_centrality': nx.closeness_centrality(self.G),
            'eigenvector_centrality': nx.eigenvector_centrality(self.G, max_iter=1000),
            'pagerank': nx.pagerank(self.G, alpha=0.85)
        }

        # 计算结构洞指标（约束度）
        constraint_dict = nx.constraint(self.G)
        self.social_metrics['structural_holes'] = constraint_dict

        # 计算核心-边缘结构（k-core分解）
        core_numbers = nx.core_number(self.G)
        self.social_metrics['k_core'] = core_numbers

        # 计算网络效率
        self.social_metrics['efficiency'] = nx.global_efficiency(self.G)

        return self.social_metrics

    def analyze_risk_distribution(self):
        """分析风险等级分布"""
        print("正在分析风险等级分布...")

        risk_stats = {}

        # 风险等级统计
        risk_counts = self.nodes_df['risk_level'].value_counts()
        risk_stats['risk_distribution'] = risk_counts.to_dict()

        # 不同风险等级的网络特征
        for risk_level in ['高', '中', '低']:
            risk_nodes = self.nodes_df[self.nodes_df['risk_level'] == risk_level]['node_id'].tolist()
            subgraph = self.G.subgraph(risk_nodes)

            if len(risk_nodes) > 1:
                risk_stats[f'{risk_level}风险子图'] = {
                    'num_nodes': len(risk_nodes),
                    'num_edges': subgraph.number_of_edges(),
                    'density': nx.density(subgraph) if len(risk_nodes) > 1 else 0,
                    'average_clustering': nx.average_clustering(subgraph)
                }
            else:
                risk_stats[f'{risk_level}风险子图'] = {
                    'num_nodes': len(risk_nodes),
                    'num_edges': 0,
                    'density': 0,
                    'average_clustering': 0
                }

        return risk_stats

    def create_visualizations(self, output_dir="visualizations"):
        """创建可视化图表"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在创建可视化图表...")

        # 1. 度分布直方图
        self.plot_degree_distribution(output_dir)

        # 2. 风险等级分布饼图
        self.plot_risk_distribution(output_dir)

        # 3. 中心性对比散点图
        self.plot_centrality_comparison(output_dir)

        # 4. 网络拓扑可视化
        self.plot_network_topology(output_dir)

        # 5. 诈骗类型分布
        self.plot_fraud_type_distribution(output_dir)

        # 6. 地理位置分布
        self.plot_location_distribution(output_dir)

    def plot_degree_distribution(self, output_dir):
        """绘制度分布"""
        degrees = [d for n, d in self.G.degree()]

        plt.figure(figsize=(12, 8))

        # 度分布直方图
        plt.subplot(2, 2, 1)
        plt.hist(degrees, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('节点度数')
        plt.ylabel('频次')
        plt.title('节点度分布直方图')
        plt.grid(True, alpha=0.3)

        # 累积度分布
        plt.subplot(2, 2, 2)
        sorted_degrees = sorted(degrees, reverse=True)
        cumsum = np.cumsum(sorted_degrees) / sum(sorted_degrees)
        plt.plot(sorted_degrees, cumsum, 'ro-', markersize=3, alpha=0.7)
        plt.xlabel('度数')
        plt.ylabel('累积概率')
        plt.title('累积度分布')
        plt.grid(True, alpha=0.3)

        # 度-度相关性
        plt.subplot(2, 2, 3)
        degrees_list = list(dict(self.G.degree()).values())
        if len(degrees_list) > 100:  # 只采样部分点以提高性能
            sample_indices = np.random.choice(len(degrees_list), size=min(100, len(degrees_list)), replace=False)
            sample_degrees = [degrees_list[i] for i in sample_indices]
            plt.scatter(sample_degrees, sample_degrees, alpha=0.6, color='green')
        plt.xlabel('节点度数')
        plt.ylabel('邻居平均度数')
        plt.title('度-度相关性')
        plt.grid(True, alpha=0.3)

        # 对数-对数度分布
        plt.subplot(2, 2, 4)
        degree_counts = Counter(degrees)
        deg, cnt = zip(*sorted(degree_counts.items()))
        plt.loglog(deg, cnt, 'bo-', markersize=4, alpha=0.7)
        plt.xlabel('度数 (log)')
        plt.ylabel('频次 (log)')
        plt.title('对数-对数度分布')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/degree_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_risk_distribution(self, output_dir):
        """绘制风险等级分布"""
        risk_counts = self.nodes_df['risk_level'].value_counts()

        plt.figure(figsize=(12, 5))

        # 饼图
        plt.subplot(1, 2, 1)
        colors = ['red', 'orange', 'green']
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('风险等级分布')

        # 条形图
        plt.subplot(1, 2, 2)
        risk_counts.plot(kind='bar', color=colors)
        plt.xlabel('风险等级')
        plt.ylabel('节点数量')
        plt.title('风险等级统计')
        plt.xticks(rotation=0)

        for i, v in enumerate(risk_counts.values):
            plt.text(i, v + 5, str(v), ha='center')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/risk_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_centrality_comparison(self, output_dir):
        """绘制中心性对比"""
        # 获取前20个高中心性节点
        top_n = 20

        # 提取各中心性度量
        degree_cent = list(self.social_metrics['degree_centrality'].values())
        between_cent = list(self.social_metrics['betweenness_centrality'].values())
        close_cent = list(self.social_metrics['closeness_centrality'].values())
        eigen_cent = list(self.social_metrics['eigenvector_centrality'].values())

        plt.figure(figsize=(15, 10))

        # 散点图矩阵
        metrics_data = {
            '度中心性': degree_cent,
            '介数中心性': between_cent,
            '接近中心性': close_cent,
            '特征向量中心性': eigen_cent
        }

        metric_names = list(metrics_data.keys())
        n_metrics = len(metric_names)

        for i in range(n_metrics):
            for j in range(n_metrics):
                plt.subplot(n_metrics, n_metrics, i*n_metrics + j + 1)

                if i == j:
                    # 对角线：直方图
                    plt.hist(metrics_data[metric_names[i]], bins=20, alpha=0.7, color='skyblue')
                    plt.title(f'{metric_names[i]}分布')
                else:
                    # 散点图
                    plt.scatter(metrics_data[metric_names[j]], metrics_data[metric_names[i]],
                              alpha=0.6, s=10, color='red')
                    if i == n_metrics-1:
                        plt.xlabel(metric_names[j])
                    if j == 0:
                        plt.ylabel(metric_names[i])

                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/centrality_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_network_topology(self, output_dir):
        """绘制网络拓扑"""
        plt.figure(figsize=(14, 10))

        # 使用spring布局
        pos = nx.spring_layout(self.G, k=0.1, iterations=50, seed=42)

        # 节点大小基于度数
        degrees = dict(self.G.degree())
        node_sizes = [degrees[node] * 20 + 50 for node in self.G.nodes()]

        # 节点颜色基于风险等级
        risk_colors = {'高': 'red', '中': 'orange', '低': 'green'}
        node_colors = [risk_colors[self.nodes_df[self.nodes_df['node_id'] == node]['risk_level'].iloc[0]]
                      for node in self.G.nodes()]

        # 边颜色基于关系类型
        edge_colors = []
        relation_colors = {'通信': 'blue', '资金': 'red', '协作': 'green'}
        for u, v in self.G.edges():
            edge_data = self.G.get_edge_data(u, v)
            edge_colors.append(relation_colors.get(edge_data.get('relation_type', '通信'), 'gray'))

        # 绘制网络
        nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes, node_color=node_colors,
                              alpha=0.7, edgecolors='black', linewidths=0.5)
        nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, alpha=0.3, width=0.5)

        # 图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='高风险'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='中风险'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='低风险'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='通信关系'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='资金关系'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='协作关系')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.title('电信诈骗犯罪网络拓扑结构', fontsize=16, fontweight='bold')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/network_topology.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_fraud_type_distribution(self, output_dir):
        """绘制诈骗类型分布"""
        fraud_counts = self.nodes_df['fraud_type'].value_counts()

        plt.figure(figsize=(12, 6))

        # 水平条形图
        bars = plt.barh(fraud_counts.index, fraud_counts.values,
                       color=sns.color_palette("husl", len(fraud_counts)))

        plt.xlabel('节点数量')
        plt.ylabel('诈骗类型')
        plt.title('诈骗类型分布')

        # 添加数值标签
        for bar, value in zip(bars, fraud_counts.values):
            plt.text(value + 1, bar.get_y() + bar.get_height()/2,
                    f'{value}', va='center', fontsize=10)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fraud_type_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_location_distribution(self, output_dir):
        """绘制地理位置分布"""
        location_counts = self.nodes_df['location'].value_counts()

        plt.figure(figsize=(12, 6))

        # 条形图
        bars = plt.bar(location_counts.index, location_counts.values,
                      color=sns.color_palette("Set3", len(location_counts)))

        plt.xlabel('地理位置')
        plt.ylabel('节点数量')
        plt.title('嫌疑人地理位置分布')
        plt.xticks(rotation=45)

        # 添加数值标签
        for bar, value in zip(bars, location_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 1,
                    f'{value}', ha='center', fontsize=10)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/location_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def export_gephi_data(self, output_dir="gephi_data"):
        """导出Gephi格式数据"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在导出Gephi数据...")

        # 导出节点数据（包含中心性指标）
        nodes_gephi = self.nodes_df.copy()

        # 添加中心性指标
        for centrality_name, centrality_dict in self.social_metrics.items():
            if centrality_name in ['degree_centrality', 'betweenness_centrality',
                                 'closeness_centrality', 'eigenvector_centrality', 'pagerank']:
                nodes_gephi[centrality_name] = nodes_gephi['node_id'].map(centrality_dict)

        nodes_gephi.to_csv(f"{output_dir}/nodes_gephi.csv", index=False, encoding='utf-8-sig')

        # 导出边数据
        edges_gephi = self.edges_df.copy()
        edges_gephi.to_csv(f"{output_dir}/edges_gephi.csv", index=False, encoding='utf-8-sig')

        print(f"Gephi数据已导出到 {output_dir} 目录")

    def generate_analysis_report(self, output_dir="analysis_results"):
        """生成分析报告"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在生成分析报告...")

        with open(f"{output_dir}/network_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write("=== 电信诈骗犯罪网络分析报告 ===\n\n")

            # 基本指标
            f.write("1. 网络基本指标:\n")
            for key, value in self.basic_metrics.items():
                if key != 'degree_stats':
                    f.write(f"   {key}: {value}\n")
                else:
                    f.write("   度统计:\n")
                    for stat_key, stat_value in value.items():
                        f.write(f"      {stat_key}: {stat_value}\n")

            # 社会网络指标统计
            f.write("\n2. 社会网络指标统计:\n")
            centrality_stats = {}
            for centrality_name in ['degree_centrality', 'betweenness_centrality',
                                  'closeness_centrality', 'eigenvector_centrality', 'pagerank']:
                values = list(self.social_metrics[centrality_name].values())
                centrality_stats[centrality_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': max(values),
                    'min': min(values)
                }

            for name, stats in centrality_stats.items():
                f.write(f"   {name}:\n")
                for stat_name, value in stats.items():
                    f.write(f"      {stat_name}: {value:.6f}\n")

            # 风险分析
            risk_stats = self.analyze_risk_distribution()
            f.write("\n3. 风险等级分析:\n")
            for key, value in risk_stats.items():
                f.write(f"   {key}: {value}\n")

        print(f"分析报告已保存到 {output_dir}/network_analysis_report.txt")

    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("=== 开始电信诈骗犯罪网络分析 ===\n")

        # 1. 加载数据
        if not self.load_data():
            return False

        # 2. 计算基础指标
        self.calculate_basic_metrics()

        # 3. 计算社会网络指标
        self.calculate_social_network_metrics()

        # 4. 创建可视化
        self.create_visualizations()

        # 5. 导出Gephi数据
        self.export_gephi_data()

        # 6. 生成分析报告
        self.generate_analysis_report()

        print("\n=== 网络分析完成 ===")
        print("生成的文件:")
        print("- visualizations/ 目录：可视化图表")
        print("- gephi_data/ 目录：Gephi导入数据")
        print("- analysis_results/ 目录：分析报告")

        return True


def main():
    """主函数"""
    analyzer = TelecomFraudNetworkAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
