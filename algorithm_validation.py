"""
电信诈骗犯罪网络关键节点识别算法验证
通过真实标签对照和网络破坏实验验证算法效果
基于复杂网络与社会网络特性的电信诈骗犯罪网络关键节点识别研究
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AlgorithmValidator:
    """算法验证器"""

    def __init__(self, data_dir="data"):
        """
        初始化验证器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.G = None
        self.nodes_df = None
        self.edges_df = None

        # 验证结果存储
        self.ground_truth = {}
        self.algorithm_predictions = {}
        self.validation_results = {}

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

    def create_ground_truth_labels(self):
        """
        创建真实标签（基于领域知识的模拟标签）
        在实际应用中，这些标签应该来自专家标注或历史数据
        """
        print("正在创建真实标签...")

        # 基于多种指标的综合判断来生成模拟真实标签
        # 1. 风险等级：高风险的节点更可能是关键节点
        # 2. 网络中心性：中心性高的节点更可能是关键节点
        # 3. 连接模式：连接多种关系类型的节点更可能是关键节点

        # 计算各种中心性
        degree_cent = nx.degree_centrality(self.G)
        betweenness_cent = nx.betweenness_centrality(self.G)
        eigenvector_cent = nx.eigenvector_centrality(self.G, max_iter=1000)

        # 计算综合评分
        key_node_scores = {}
        risk_mapping = {'高': 3, '中': 2, '低': 1}

        for node in self.G.nodes():
            # 风险等级得分
            node_data = self.nodes_df[self.nodes_df['node_id'] == node]
            if not node_data.empty:
                risk_level = node_data['risk_level'].iloc[0]
                risk_score = risk_mapping.get(risk_level, 1)
            else:
                risk_score = 1

            # 中心性得分（归一化后平均）
            cent_scores = [
                degree_cent.get(node, 0),
                betweenness_cent.get(node, 0),
                eigenvector_cent.get(node, 0)
            ]
            avg_cent_score = np.mean(cent_scores)

            # 连接多样性得分（连接不同关系类型的边数）
            edges_from_node = self.edges_df[
                (self.edges_df['source'] == node) | (self.edges_df['target'] == node)
            ]
            unique_relations = edges_from_node['relation_type'].nunique()
            diversity_score = unique_relations / 3.0  # 归一化到0-1

            # 综合评分
            combined_score = (risk_score * 0.4 + avg_cent_score * 0.4 + diversity_score * 0.2)

            key_node_scores[node] = combined_score

        # 基于综合评分生成二分类标签
        # Top 20% 的节点被认为是关键节点
        sorted_scores = sorted(key_node_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = int(len(sorted_scores) * 0.2)  # 20% 作为关键节点

        ground_truth = {}
        for i, (node, score) in enumerate(sorted_scores):
            ground_truth[node] = 1 if i < top_k else 0

        self.ground_truth = ground_truth
        print(f"生成真实标签完成：{sum(ground_truth.values())}个关键节点，{len(ground_truth) - sum(ground_truth.values())}个普通节点")

        return ground_truth

    def load_algorithm_predictions(self):
        """加载算法预测结果"""
        print("正在加载算法预测结果...")

        # 这里我们重新计算各种算法的结果
        # 在实际应用中，这些结果应该从之前保存的文件中加载

        algorithms = {
            '度中心性': nx.degree_centrality(self.G),
            '介数中心性': nx.betweenness_centrality(self.G),
            '接近中心性': nx.closeness_centrality(self.G),
            '特征向量中心性': nx.eigenvector_centrality(self.G, max_iter=1000),
            'PageRank': nx.pagerank(self.G, alpha=0.85)
        }

        # 转换为0-1范围的预测概率
        self.algorithm_predictions = {}
        for alg_name, scores in algorithms.items():
            # 归一化到0-1范围
            score_values = list(scores.values())
            min_score = min(score_values)
            max_score = max(score_values)

            if max_score > min_score:
                normalized_scores = {node: (score - min_score) / (max_score - min_score)
                                   for node, score in scores.items()}
            else:
                normalized_scores = {node: 0.5 for node in scores.keys()}

            self.algorithm_predictions[alg_name] = normalized_scores

        print(f"加载了 {len(self.algorithm_predictions)} 个算法的预测结果")

        return self.algorithm_predictions

    def evaluate_classification_performance(self):
        """评估分类性能"""
        print("正在评估分类性能...")

        if not self.ground_truth or not self.algorithm_predictions:
            print("真实标签或预测结果未准备好")
            return {}

        results = {}

        for alg_name, predictions in self.algorithm_predictions.items():
            # 获取预测概率和真实标签
            y_true = []
            y_scores = []

            for node in self.G.nodes():
                if node in self.ground_truth and node in predictions:
                    y_true.append(self.ground_truth[node])
                    y_scores.append(predictions[node])

            if not y_true:
                continue

            y_true = np.array(y_true)
            y_scores = np.array(y_scores)

            # 计算ROC曲线和AUC
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            # 计算Precision-Recall曲线
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)

            # 基于默认阈值(0.5)的分类性能
            y_pred = (y_scores >= 0.5).astype(int)

            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0

            results[alg_name] = {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'accuracy': accuracy,
                'precision': precision_score,
                'recall': recall_score,
                'f1_score': f1_score,
                'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
                'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
                'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
            }

        self.validation_results['classification'] = results
        print("分类性能评估完成")

        return results

    def perform_network_disruption_analysis(self):
        """执行网络破坏实验"""
        print("正在执行网络破坏实验...")

        # 选择要测试的算法
        test_algorithms = {
            '随机移除': None,  # 随机选择节点
            '按度数移除': nx.degree_centrality(self.G),
            '按介数中心性移除': nx.betweenness_centrality(self.G),
            '按特征向量中心性移除': nx.eigenvector_centrality(self.G, max_iter=1000)
        }

        disruption_results = {}

        # 分析函数
        def calculate_network_metrics(graph):
            """计算网络关键指标"""
            if graph.number_of_nodes() == 0:
                return {'components': 0, 'efficiency': 0, 'coverage': 0}

            # 最大连通分量占比
            largest_cc = max(nx.connected_components(graph), key=len) if graph.number_of_edges() > 0 else set()
            coverage = len(largest_cc) / graph.number_of_nodes()

            # 网络效率
            efficiency = nx.global_efficiency(graph)

            # 连通分量数
            components = nx.number_connected_components(graph)

            return {
                'components': components,
                'efficiency': efficiency,
                'coverage': coverage
            }

        original_metrics = calculate_network_metrics(self.G)

        for alg_name, ranking in test_algorithms.items():
            print(f"  分析{alg_name}策略...")

            G_temp = self.G.copy()
            removed_nodes = []
            metrics_over_time = [original_metrics.copy()]

            # 移除比例从0%到50%
            remove_fractions = np.linspace(0.02, 0.5, 25)  # 移除2%到50%的节点

            for frac in remove_fractions:
                num_to_remove = int(frac * self.G.number_of_nodes()) - len(removed_nodes)

                if num_to_remove <= 0 or G_temp.number_of_nodes() == 0:
                    continue

                if ranking is None:
                    # 随机移除
                    nodes_to_remove = np.random.choice(
                        list(G_temp.nodes()),
                        size=min(num_to_remove, G_temp.number_of_nodes()),
                        replace=False
                    )
                else:
                    # 按重要性排序移除
                    remaining_nodes = list(G_temp.nodes())
                    remaining_ranking = {node: ranking.get(node, 0) for node in remaining_nodes}
                    sorted_remaining = sorted(remaining_ranking.items(), key=lambda x: x[1], reverse=True)
                    nodes_to_remove = [node for node, score in sorted_remaining[:num_to_remove]]

                G_temp.remove_nodes_from(nodes_to_remove)
                removed_nodes.extend(nodes_to_remove)

                current_metrics = calculate_network_metrics(G_temp)
                metrics_over_time.append(current_metrics)

            disruption_results[alg_name] = {
                'removed_fractions': [0] + list(remove_fractions[:len(metrics_over_time)-1]),
                'metrics_over_time': metrics_over_time
            }

        self.validation_results['disruption'] = disruption_results
        print("网络破坏实验完成")

        return disruption_results

    def create_validation_visualizations(self, output_dir="validation_results"):
        """创建验证结果可视化"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在创建验证结果可视化...")

        # 1. ROC曲线对比
        self.plot_roc_curves(output_dir)

        # 2. Precision-Recall曲线对比
        self.plot_pr_curves(output_dir)

        # 3. 分类性能对比条形图
        self.plot_classification_metrics(output_dir)

        # 4. 网络破坏实验结果
        self.plot_disruption_analysis(output_dir)

        # 5. 综合性能雷达图
        self.plot_algorithm_radar_comparison(output_dir)

    def plot_roc_curves(self, output_dir):
        """绘制ROC曲线对比"""
        if 'classification' not in self.validation_results:
            return

        plt.figure(figsize=(10, 8))

        for alg_name, results in self.validation_results['classification'].items():
            roc_data = results['roc_curve']
            plt.plot(roc_data['fpr'], roc_data['tpr'],
                    label=f'{alg_name} (AUC = {results["roc_auc"]:.3f})',
                    linewidth=2, alpha=0.8)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机猜测')
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('关键节点识别算法ROC曲线对比')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/roc_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pr_curves(self, output_dir):
        """绘制Precision-Recall曲线对比"""
        if 'classification' not in self.validation_results:
            return

        plt.figure(figsize=(10, 8))

        for alg_name, results in self.validation_results['classification'].items():
            pr_data = results['pr_curve']
            plt.plot(pr_data['recall'], pr_data['precision'],
                    label=f'{alg_name} (AUC = {results["pr_auc"]:.3f})',
                    linewidth=2, alpha=0.8)

        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title('关键节点识别算法Precision-Recall曲线对比')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/pr_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_classification_metrics(self, output_dir):
        """绘制分类性能对比条形图"""
        if 'classification' not in self.validation_results:
            return

        plt.figure(figsize=(15, 10))

        algorithms = list(self.validation_results['classification'].keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        metric_names = ['准确率', '精确率', '召回率', 'F1得分', 'ROC-AUC', 'PR-AUC']

        # 为每个指标创建子图
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            plt.subplot(2, 3, i+1)

            values = [self.validation_results['classification'][alg][metric] for alg in algorithms]
            bars = plt.bar(range(len(algorithms)), values, alpha=0.7, color='skyblue', edgecolor='black')

            plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
            plt.ylabel(name)
            plt.title(f'{name}对比')

            # 添加数值标签
            for j, (bar, value) in enumerate(zip(bars, values)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        '.3f', ha='center', va='bottom', fontsize=8)

            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/classification_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_disruption_analysis(self, output_dir):
        """绘制网络破坏实验结果"""
        if 'disruption' not in self.validation_results:
            return

        disruption = self.validation_results['disruption']

        plt.figure(figsize=(15, 10))

        # 子图1：连通分量数变化
        plt.subplot(2, 2, 1)
        for alg_name, results in disruption.items():
            fractions = results['removed_fractions']
            components = [m['components'] for m in results['metrics_over_time']]
            plt.plot(fractions, components, label=alg_name, linewidth=2, marker='o', markersize=3)

        plt.xlabel('移除节点比例')
        plt.ylabel('连通分量数')
        plt.title('网络破坏对连通性的影响')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：最大连通分量占比变化
        plt.subplot(2, 2, 2)
        for alg_name, results in disruption.items():
            fractions = results['removed_fractions']
            coverage = [m['coverage'] for m in results['metrics_over_time']]
            plt.plot(fractions, coverage, label=alg_name, linewidth=2, marker='s', markersize=3)

        plt.xlabel('移除节点比例')
        plt.ylabel('最大连通分量占比')
        plt.title('网络破坏对覆盖率的影响')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图3：网络效率变化
        plt.subplot(2, 2, 3)
        for alg_name, results in disruption.items():
            fractions = results['removed_fractions']
            efficiency = [m['efficiency'] for m in results['metrics_over_time']]
            plt.plot(fractions, efficiency, label=alg_name, linewidth=2, marker='^', markersize=3)

        plt.xlabel('移除节点比例')
        plt.ylabel('网络效率')
        plt.title('网络破坏对效率的影响')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图4：破坏策略效果对比
        plt.subplot(2, 2, 4)
        strategies = list(disruption.keys())
        final_coverage = [disruption[strategy]['metrics_over_time'][-1]['coverage'] for strategy in strategies]

        bars = plt.bar(strategies, final_coverage, color=['gray', 'red', 'blue', 'green'], alpha=0.7, edgecolor='black')
        plt.ylabel('移除50%节点后的覆盖率')
        plt.title('不同破坏策略的最终效果')
        plt.xticks(rotation=45)

        for bar, value in zip(bars, final_coverage):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.3f', ha='center', va='bottom')

        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/disruption_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_algorithm_radar_comparison(self, output_dir):
        """绘制算法雷达对比图"""
        if 'classification' not in self.validation_results:
            return

        plt.figure(figsize=(12, 8))

        # 准备雷达图数据
        algorithms = list(self.validation_results['classification'].keys())
        categories = ['准确率', '精确率', '召回率', 'F1得分', 'ROC-AUC']
        category_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

        # 计算角度
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]  # 闭合雷达图

        ax = plt.subplot(111, polar=True)

        for alg_name in algorithms:
            values = [self.validation_results['classification'][alg_name][key] for key in category_keys]
            values += values[:1]  # 闭合

            ax.plot(angles, values, 'o-', linewidth=2, label=alg_name, markersize=6, alpha=0.8)

        # 添加网格线
        ax.set_thetagrids([a * 180/np.pi for a in angles[:-1]], categories)

        # 设置标题
        ax.set_title('关键节点识别算法综合性能雷达图', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 添加参考线（平均性能）
        avg_values = []
        for key in category_keys:
            avg_val = np.mean([self.validation_results['classification'][alg][key] for alg in algorithms])
            avg_values.append(avg_val)
        avg_values += avg_values[:1]

        ax.plot(angles, avg_values, 'k--', linewidth=1, label='平均性能', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/algorithm_radar_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_validation_report(self, output_dir="validation_results"):
        """生成验证报告"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在生成算法验证报告...")

        with open(f"{output_dir}/algorithm_validation_report.txt", 'w', encoding='utf-8') as f:
            f.write("=== 电信诈骗犯罪网络关键节点识别算法验证报告 ===\n\n")

            # 1. 真实标签创建说明
            f.write("1. 真实标签创建说明:\n")
            f.write("   基于以下标准生成模拟真实标签:\n")
            f.write("   • 风险等级：高风险节点更可能是关键节点\n")
            f.write("   • 网络中心性：中心性高的节点更可能是关键节点\n")
            f.write("   • 连接多样性：连接多种关系类型的节点更可能是关键节点\n")
            key_nodes_count = sum(self.ground_truth.values()) if self.ground_truth else 0
            total_nodes = len(self.ground_truth) if self.ground_truth else 0
            f.write(f"   • 关键节点数量: {key_nodes_count} ({key_nodes_count/total_nodes*100:.1f}%)\n\n")

            # 2. 分类性能评估
            if 'classification' in self.validation_results:
                f.write("2. 分类性能评估:\n")
                for alg_name, results in self.validation_results['classification'].items():
                    f.write(f"   {alg_name}:\n")
                    f.write(f"      准确率: {results['accuracy']:.4f}\n")
                    f.write(f"      精确率: {results['precision']:.4f}\n")
                    f.write(f"      召回率: {results['recall']:.4f}\n")
                    f.write(f"      F1得分: {results['f1_score']:.4f}\n")
                    f.write(f"      ROC-AUC: {results['roc_auc']:.4f}\n")
                    f.write(f"      PR-AUC: {results['pr_auc']:.4f}\n")
                    cm = results['confusion_matrix']
                    f.write(f"      混淆矩阵: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}\n")
                f.write("\n")

                # 找出最佳算法
                best_alg = max(self.validation_results['classification'].items(),
                              key=lambda x: x[1]['f1_score'])
                f.write(f"   最佳算法: {best_alg[0]} (F1得分: {best_alg[1]['f1_score']:.4f})\n\n")

            # 3. 网络破坏实验结果
            if 'disruption' in self.validation_results:
                f.write("3. 网络破坏实验结果:\n")

                disruption = self.validation_results['disruption']

                # 分析不同策略的效果
                f.write("   破坏策略效果分析:\n")
                for strategy, results in disruption.items():
                    final_metrics = results['metrics_over_time'][-1]
                    f.write(f"      {strategy}:\n")
                    f.write(f"         移除50%节点后连通分量数: {final_metrics['components']}\n")
                    f.write(f"         最大连通分量占比: {final_metrics['coverage']:.4f}\n")
                    f.write(f"         网络效率: {final_metrics['efficiency']:.4f}\n")

                f.write("\n   实验结论:\n")
                # 比较随机移除和针对性移除的效果
                random_final = disruption['随机移除']['metrics_over_time'][-1]['coverage']
                targeted_final = disruption['按介数中心性移除']['metrics_over_time'][-1]['coverage']

                if targeted_final < random_final:
                    f.write("   ✓ 针对性移除比随机移除更有效，能更快破坏网络结构\n")
                else:
                    f.write("   ⚠ 网络对针对性攻击的鲁棒性较强\n")

                f.write("\n")

            # 4. 算法验证结论
            f.write("4. 算法验证结论:\n")

            if 'classification' in self.validation_results:
                avg_f1 = np.mean([r['f1_score'] for r in self.validation_results['classification'].values()])
                avg_auc = np.mean([r['roc_auc'] for r in self.validation_results['classification'].values()])

                f.write(f"   • 整体性能: 平均F1得分={avg_f1:.4f}, 平均ROC-AUC={avg_auc:.4f}\n")

                if avg_auc > 0.8:
                    f.write("   • 算法有效性: 优秀，算法能够有效识别关键节点\n")
                elif avg_auc > 0.7:
                    f.write("   • 算法有效性: 良好，算法具有一定的识别能力\n")
                else:
                    f.write("   • 算法有效性: 一般，需要进一步优化\n")

            f.write("\n   实际应用建议:\n")
            f.write("   • 优先使用综合性能最好的算法进行关键节点识别\n")
            f.write("   • 结合网络破坏实验结果，选择对网络破坏效果最好的策略\n")
            f.write("   • 在实际应用中，应结合领域专家知识调整算法参数\n")
            f.write("   • 定期重新评估算法性能，适应网络结构的变化\n")

        print(f"算法验证报告已保存到 {output_dir}/algorithm_validation_report.txt")

    def run_complete_validation(self):
        """运行完整验证流程"""
        print("=== 开始电信诈骗犯罪网络关键节点识别算法验证 ===\n")

        if not self.load_data():
            return False

        # 1. 创建真实标签
        self.create_ground_truth_labels()

        # 2. 加载算法预测结果
        self.load_algorithm_predictions()

        # 3. 评估分类性能
        self.evaluate_classification_performance()

        # 4. 执行网络破坏实验
        self.perform_network_disruption_analysis()

        # 5. 创建可视化
        self.create_validation_visualizations()

        # 6. 生成验证报告
        self.generate_validation_report()

        print("\n=== 算法验证完成 ===")
        print("生成的文件:")
        print("- validation_results/ 目录：验证结果可视化图表")
        print("- validation_results/algorithm_validation_report.txt：详细验证报告")

        return True


def main():
    """主函数"""
    validator = AlgorithmValidator()
    validator.run_complete_validation()


if __name__ == "__main__":
    main()
