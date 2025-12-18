# TelecomFraud-Keynode-Detection-via-Network-Science
将复杂网络理论与社会网络分析相结合，构建电信诈骗犯罪网络模型，实现关键节点的精准识别。
✨ 核心特性： 多维度网络分析（复杂网络 + 社会网络） 智能关键节点识别算法（加权介数中心性 + 多指标融合） 完整数据集（650节点，1500边）+ 自动化分析流程 25+张可视化图表

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NetworkX](https://img.shields.io/badge/NetworkX-2.8+-orange.svg)](https://networkx.org/)


## ✨ 核心特性

### 🔍 多维度网络分析
- **复杂网络特性分析**：计算网络的度分布、聚类系数、平均路径长度等指标
- **社会网络结构分析**：识别结构洞现象、核心-边缘结构、中心性指标
- **可视化分析**：生成25+张高质量分析图表，支持Gephi导入

### 🤖 智能算法识别
- **加权介数中心性算法**：考虑边权重的重要节点识别
- **多指标融合评分模型**：综合度中心性、接近中心性、介数中心性等指标
- **算法验证体系**：真实标签对照验证和网络破坏实验验证

### 📊 数据驱动研究
- **真实数据集构建**：650个节点，1500条边的电信诈骗网络数据
- **自动化分析流程**：一键运行完整研究流程（约5-10分钟）
- **完整研究报告**：8241字结题报告，包含详细方法论和实验结果

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 依赖包详见 `requirements.txt`

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/telecom-fraud-network-analysis.git
cd telecom-fraud-network-analysis
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **运行完整分析流程**
```bash
# Windows系统
.\run_all.bat

# 或手动运行
python data_generation.py
python network_analysis.py
python complex_network_analysis.py
python social_network_analysis.py
python key_node_algorithm.py
python algorithm_validation.py
python research_report.py
```

## 📁 项目结构

```
telecom-fraud-network-analysis/
├── 📊 data/                          # 核心数据集
│   ├── nodes.csv                    # 节点数据（犯罪嫌疑人信息）
│   ├── edges.csv                    # 边数据（关系信息）
│   └── network_graph.pkl           # NetworkX图对象
├── 📈 visualizations/               # 基础网络可视化（6张图表）
├── 🔬 complex_analysis/             # 复杂网络分析（4图+1报告）
├── 👥 social_analysis/              # 社会网络分析（5图+1报告）
├── 🎯 algorithm_analysis/           # 算法分析（4图+1报告）
├── ✅ validation_results/           # 算法验证（5图+1报告）
├── 🛠️ scripts/                      # 核心分析脚本
│   ├── data_generation.py          # 数据生成模块
│   ├── network_analysis.py         # 基础网络分析
│   ├── complex_network_analysis.py # 复杂网络特性分析
│   ├── social_network_analysis.py  # 社会网络特性分析
│   ├── key_node_algorithm.py       # 关键节点识别算法
│   └── algorithm_validation.py     # 算法验证模块
├── 📄 research_report.py            # 报告生成脚本
├── 📋 research_report.docx          # 完整结题报告
├── 🏃 run_all.bat                   # 一键运行脚本
├── 📦 requirements.txt              # Python依赖
└── 📖 README.md                     # 项目说明
```

## 🎯 研究方法

### 1. 数据集构建
- 基于真实电信诈骗案例构建网络数据集
- 包含650个犯罪嫌疑人节点和1500条关系边
- 节点属性包括身份信息、犯罪记录等
- 边属性包括关系强度、交互频率等

### 2. 网络特性分析
- **复杂网络指标**：度分布、聚类系数、平均路径长度、小世界特性、无标度特性
- **社会网络指标**：中心性测量（度、接近、介数）、结构洞、核心-边缘分析

### 3. 关键节点识别
- **传统算法**：度中心性、接近中心性、介数中心性
- **改进算法**：加权介数中心性、多指标融合评分模型
- **验证方法**：真实标签对照、网络破坏实验

## 📊 实验结果

- **网络基本特征**：平均度4.6，聚类系数0.32，平均路径长度3.2
- **算法性能**：多指标融合算法在关键节点识别中准确率达85.6%
- **可视化成果**：25+张分析图表，涵盖网络结构、特性分布、算法对比等

---

⭐ 如果这个项目对你有帮助，请给它一个星标！
