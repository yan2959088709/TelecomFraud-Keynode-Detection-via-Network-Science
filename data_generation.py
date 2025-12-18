"""
电信诈骗犯罪网络数据集构建脚本
基于复杂网络与社会网络特性的电信诈骗犯罪网络关键节点识别研究
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import networkx as nx


class TelecomFraudNetworkGenerator:
    """电信诈骗网络数据生成器"""

    def __init__(self, num_nodes=650, num_edges=1500):
        """
        初始化网络生成器

        Args:
            num_nodes: 节点数量（犯罪嫌疑人）
            num_edges: 边数量（关系连接）
        """
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        # 节点属性定义
        self.node_attributes = {
            'node_id': [],  # 嫌疑人ID
            'risk_level': [],  # 风险等级 (高/中/低)
            'fraud_amount': [],  # 涉案金额
            'fraud_type': [],  # 诈骗类型
            'location': [],  # 地理位置
            'age_group': [],  # 年龄段
            'education_level': []  # 教育水平
        }

        # 边属性定义
        self.edge_attributes = {
            'source': [],  # 源节点
            'target': [],  # 目标节点
            'relation_type': [],  # 关系类型 (通信/资金/协作)
            'weight': [],  # 关系权重
            'frequency': [],  # 联系频率
            'duration': [],  # 持续时间（天）
            'amount': []  # 资金金额（针对资金流向）
        }

    def generate_nodes(self):
        """生成节点数据"""
        print(f"正在生成 {self.num_nodes} 个犯罪嫌疑人节点...")

        # 风险等级分布 (高:30%, 中:50%, 低:20%)
        risk_levels = ['高', '中', '低']
        risk_weights = [0.3, 0.5, 0.2]

        # 诈骗类型
        fraud_types = ['网络诈骗', '电话诈骗', '短信诈骗', '投资理财诈骗', '兼职诈骗', '色情诈骗']

        # 地理位置
        locations = ['北京', '上海', '广州', '深圳', '杭州', '南京', '苏州', '武汉', '成都', '重庆']

        # 年龄段
        age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']

        # 教育水平
        education_levels = ['初中及以下', '高中', '大专', '本科', '研究生及以上']

        for i in range(1, self.num_nodes + 1):
            # 生成节点ID
            self.node_attributes['node_id'].append(f"S{i:04d}")

            # 随机选择风险等级
            self.node_attributes['risk_level'].append(
                np.random.choice(risk_levels, p=risk_weights)
            )

            # 涉案金额（根据风险等级调整）
            risk_multiplier = {'高': 3, '中': 1.5, '低': 0.5}
            base_amount = np.random.lognormal(mean=10, sigma=1) * 1000  # 对数正态分布
            self.node_attributes['fraud_amount'].append(
                round(base_amount * risk_multiplier[self.node_attributes['risk_level'][-1]], 2)
            )

            # 其他属性
            self.node_attributes['fraud_type'].append(np.random.choice(fraud_types))
            self.node_attributes['location'].append(np.random.choice(locations))
            self.node_attributes['age_group'].append(np.random.choice(age_groups))
            self.node_attributes['education_level'].append(np.random.choice(education_levels))

    def generate_edges(self):
        """生成边数据"""
        print(f"正在生成 {self.num_edges} 条关系边...")

        # 关系类型及其权重
        relation_types = ['通信', '资金', '协作']
        relation_weights = [0.5, 0.3, 0.2]  # 通信关系最常见

        # 生成无向边（避免重复）
        generated_edges = set()

        while len(generated_edges) < self.num_edges:
            # 随机选择两个不同节点
            source = f"S{random.randint(1, self.num_nodes):04d}"
            target = f"S{random.randint(1, self.num_nodes):04d}"

            # 确保源节点ID小于目标节点ID，避免重复
            if source > target:
                source, target = target, source

            edge_key = (source, target)

            # 如果边不存在，添加它
            if edge_key not in generated_edges and source != target:
                generated_edges.add(edge_key)

                # 添加边属性
                self.edge_attributes['source'].append(source)
                self.edge_attributes['target'].append(target)
                self.edge_attributes['relation_type'].append(
                    np.random.choice(relation_types, p=relation_weights)
                )

                # 根据关系类型生成权重和频率
                rel_type = self.edge_attributes['relation_type'][-1]
                if rel_type == '通信':
                    weight = np.random.uniform(0.1, 1.0)
                    freq = random.randint(1, 50)
                    amount = 0
                elif rel_type == '资金':
                    weight = np.random.uniform(0.3, 1.0)
                    freq = random.randint(1, 20)
                    amount = round(np.random.lognormal(mean=8, sigma=1), 2)
                else:  # 协作
                    weight = np.random.uniform(0.2, 0.8)
                    freq = random.randint(1, 10)
                    amount = 0

                self.edge_attributes['weight'].append(round(weight, 3))
                self.edge_attributes['frequency'].append(freq)
                self.edge_attributes['duration'].append(random.randint(1, 365))
                self.edge_attributes['amount'].append(amount)

    def create_network_graph(self):
        """创建NetworkX图对象"""
        print("正在创建网络图对象...")

        # 创建无向图
        G = nx.Graph()

        # 添加节点
        for i in range(self.num_nodes):
            node_data = {
                'risk_level': self.node_attributes['risk_level'][i],
                'fraud_amount': self.node_attributes['fraud_amount'][i],
                'fraud_type': self.node_attributes['fraud_type'][i],
                'location': self.node_attributes['location'][i],
                'age_group': self.node_attributes['age_group'][i],
                'education_level': self.node_attributes['education_level'][i]
            }
            G.add_node(self.node_attributes['node_id'][i], **node_data)

        # 添加边
        for i in range(len(self.edge_attributes['source'])):
            edge_data = {
                'relation_type': self.edge_attributes['relation_type'][i],
                'weight': self.edge_attributes['weight'][i],
                'frequency': self.edge_attributes['frequency'][i],
                'duration': self.edge_attributes['duration'][i],
                'amount': self.edge_attributes['amount'][i]
            }
            G.add_edge(
                self.edge_attributes['source'][i],
                self.edge_attributes['target'][i],
                **edge_data
            )

        return G

    def save_data(self, output_dir="data"):
        """保存数据到文件"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在保存节点和边数据...")

        # 保存节点数据
        nodes_df = pd.DataFrame(self.node_attributes)
        nodes_df.to_csv(f"{output_dir}/nodes.csv", index=False, encoding='utf-8-sig')

        # 保存边数据
        edges_df = pd.DataFrame(self.edge_attributes)
        edges_df.to_csv(f"{output_dir}/edges.csv", index=False, encoding='utf-8-sig')

        print(f"数据已保存到 {output_dir} 目录")
        print(f"节点数据: {len(nodes_df)} 行")
        print(f"边数据: {len(edges_df)} 行")

        return nodes_df, edges_df

    def get_statistics(self):
        """获取数据集统计信息"""
        print("\n=== 数据集统计信息 ===")

        # 节点统计
        nodes_df = pd.DataFrame(self.node_attributes)
        print("节点统计:")
        print(f"- 总数: {len(nodes_df)}")
        print(f"- 风险等级分布: {nodes_df['risk_level'].value_counts().to_dict()}")
        print(f"- 诈骗类型分布: {nodes_df['fraud_type'].value_counts().to_dict()}")
        print(f"- 平均涉案金额: {nodes_df['fraud_amount'].mean():.2f}元")
        print(f"- 涉案金额范围: {nodes_df['fraud_amount'].min():.2f} - {nodes_df['fraud_amount'].max():.2f}元")

        # 边统计
        edges_df = pd.DataFrame(self.edge_attributes)
        print("\n边统计:")
        print(f"- 总数: {len(edges_df)}")
        print(f"- 关系类型分布: {edges_df['relation_type'].value_counts().to_dict()}")
        print(f"- 平均权重: {edges_df['weight'].mean():.3f}")
        print(f"- 平均联系频率: {edges_df['frequency'].mean():.1f} 次")

        return nodes_df, edges_df


def main():
    """主函数"""
    print("=== 电信诈骗犯罪网络数据集构建 ===")

    # 设置随机种子保证可重现性
    np.random.seed(42)
    random.seed(42)

    # 创建生成器
    generator = TelecomFraudNetworkGenerator(num_nodes=650, num_edges=1500)

    # 生成数据
    generator.generate_nodes()
    generator.generate_edges()

    # 获取统计信息
    nodes_df, edges_df = generator.get_statistics()

    # 保存数据
    nodes_df, edges_df = generator.save_data()

    # 创建网络图
    G = generator.create_network_graph()
    print(f"\n网络图创建完成:")
    print(f"- 节点数: {G.number_of_nodes()}")
    print(f"- 边数: {G.number_of_edges()}")
    print(f"- 网络密度: {nx.density(G):.4f}")

    # 保存图对象
    import pickle
    with open('data/network_graph.pkl', 'wb') as f:
        pickle.dump(G, f)

    print("\n数据集构建完成！")


if __name__ == "__main__":
    main()
