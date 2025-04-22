# 信贷风险预测联邦学习系统 (CFLP)

CFLP (Credit Federation Learning Project) 是一个基于联邦学习的信贷风险预测系统。该系统支持标准机器学习和联邦学习两种训练模式，并实现了多种机器学习模型，包括XGBoost、随机森林、逻辑回归和神经网络等。

## 项目结构

```
CFLP/
├── data/                                    # 数据文件目录
│   └── default of credit card clients.csv   # 信用卡违约数据集
├── src/                                     # 源代码目录
│   ├── config/                             # 配置文件
│   │   └── default.yaml                    # 默认配置文件
│   ├── data_process/                       # 数据处理模块
│   │   └── generate_credit_card_data.py    # 信用卡数据处理生成类
│   ├── experiments/                        # 实验模块
│   │   ├── base_experiment.py             # 实验基类
│   │   ├── standard.py                    # 标准训练实验
│   │   └── federated.py                   # 联邦学习实验
│   ├── federation/                         # 联邦学习模块
│   │   ├── fed_client.py                  # 联邦学习客户端
│   │   └── fed_server.py                  # 联邦学习服务器
│   ├── models/                            # 模型实现
│   │   ├── __init__.py
│   │   ├── base_model.py                  # 模型基类
│   │   ├── neural_net.py                  # 神经网络模型
│   │   ├── logistic_regression.py         # 逻辑回归模型
│   │   ├── xgboost_model.py              # XGBoost模型
│   │   └── random_forest.py               # 随机森林模型
│   ├── utils/                             # 工具函数
│   │   ├── __init__.py
│   │   ├── metrics.py                     # 评估指标
│   │   └── logging_config.py              # 日志配置
│   ├── visualization/                      # 可视化模块
│   │   └── plot_utils.py                  # 绘图工具
│   └── main.py                            # 主程序入口
├── docker_federated/                       # 联邦学习Docker配置
│   ├── docker/                            # Docker配置文件
│   │   ├── docker-compose.yml             # 容器编排配置
│   │   ├── Dockerfile.client              # 客户端Docker镜像配置
│   │   └── Dockerfile.server              # 服务端Docker镜像配置
│   ├── grpc/                              # gRPC通信模块
│   │   ├── client_grpc.py                 # 客户端gRPC实现
│   │   ├── server_grpc.py                 # 服务端gRPC实现
│   │   ├── parameter_handler.py           # 参数处理工具
│   │   ├── protos/                        # 协议缓冲定义
│   │   │   └── federation.proto           # 联邦学习通信协议定义
│   │   └── generated/                     # 自动生成的gRPC代码
│   ├── certs/                             # SSL/TLS证书目录
│   │   ├── client/                        # 客户端证书
│   │   └── server/                        # 服务端证书
│   ├── scripts/                           # 辅助脚本
│   │   └── generate_grpc.py              # 生成gRPC代码的脚本
│   └── requirements.txt                   # 联邦学习环境依赖
└── README.md                              # 项目文档
```

## 主要功能

### 1. 标准机器学习实验
- 支持多种机器学习模型:
  - XGBoost
    * 基于梯度提升树的集成学习模型
    * 支持特征重要性分析
    * 内置处理缺失值机制
    * 可调整的正则化参数
  - 随机森林
    * 基于决策树的集成学习模型
    * 支持并行训练
    * 内置特征重要性评估
    * 随机特征选择减少过拟合
  - 逻辑回归
    * 基于SGD优化的实现
    * 支持自适应学习率
    * 内置网格搜索调参功能
    * **支持可配置的SMOTE过采样处理类别不平衡** (通过`use_smote`参数控制)
    * 可调整的分类阈值
  - 神经网络
    * 多层感知机结构(128-64-32-1)
    * 使用ReLU激活函数
    * 批量归一化层提高训练稳定性
    * 残差连接加速训练收敛
    * Dropout层防止过拟合
    * 支持GPU加速训练
- 模型性能评估
  * 准确率(Accuracy)
  * 精确率(Precision)
  * 召回率(Recall)
  * F1分数
  * AUC-ROC曲线
  * 混淆矩阵
- 可视化分析
  * 训练过程监控
  * 特征重要性分析
  * ROC曲线绘制
  * 预测结果分布

### 2. 联邦学习实验
- 分布式架构
  * 服务器-客户端模式
  * 支持多客户端并行训练
  * 安全的模型参数传输
- 联邦平均算法(FedAvg)实现
  * 支持异步模型更新
  * 自适应聚合权重
  * 通信效率优化
- 支持的联邦学习模型
  * 逻辑回归
    - 基于SGD的分布式优化
    - 支持L1/L2正则化
    - 可启用SMOTE过采样提升少数类性能
  * 神经网络
    - 分布式梯度下降
    - 模型参数聚合
    - 支持局部多轮训练
- 隐私保护机制
  * 本地数据不出本地
  * 仅传输模型参数
  * 支持差分隐私
- 性能优化
  * 通信压缩
  * 计算资源调度
  * 负载均衡

### 3. 数据处理功能
- 数据预处理
  * 缺失值处理
  * 异常值检测
  * 特征标准化
  * 类别特征编码
- 特征工程
  * 特征选择
  * 特征交叉
  * 特征重要性分析
- 数据集划分
  * 训练集/测试集划分
  * 交叉验证支持
  * 分层采样
- 类别不平衡处理
  * SMOTE过采样
  * 类别权重调整
  * 阈值优化

### 4. 基于Docker的联邦学习部署
- 容器化架构
  * 独立的服务端和客户端容器
  * 基于Docker Compose的多容器编排
  * 灵活的环境变量配置
- 安全通信
  * 基于gRPC的高效通信
  * TLS加密传输
  * 证书验证机制
- 参数序列化与传输
  * 高效的NumPy数组序列化
  * Protocol Buffers数据定义
  * 自动化的参数编码与解码
- 分布式训练流程
  * 客户端注册与发现
  * 同步的模型参数更新
  * 训练状态监控与恢复

## 系统架构与工作流程

### 标准机器学习模式
1. 数据处理: 读取原始信用卡数据，进行清洗、特征工程和训练/测试集划分
2. 模型训练: 使用训练集训练多种机器学习模型
3. 模型评估: 使用测试集评估模型性能，计算各项指标
4. 结果分析: 比较不同模型的性能，进行特征重要性分析

### 联邦学习模式
1. 数据分割: 将数据集分割为多个子集，分别分发给不同的客户端
2. 初始化: 服务端初始化全局模型，客户端加载本地数据
3. 联邦训练循环:
   - 本地训练: 各客户端使用本地数据训练模型
   - 参数上传: 客户端将模型参数上传至服务端
   - 参数聚合: 服务端聚合所有客户端参数
   - 参数分发: 服务端将全局模型参数下发给客户端
4. 全局评估: 使用全局测试集评估最终模型性能

### gRPC通信流程
1. 客户端注册: 客户端启动后向服务端注册，获取初始模型参数
2. 训练状态检查: 客户端定期检查当前训练状态，等待所有客户端就绪
3. 参数提交: 完成本地训练后，客户端将模型参数提交给服务端
4. 全局更新: 服务端聚合参数并通知客户端获取更新后的全局模型
5. 训练完成: 达到指定轮次后，服务端通知客户端训练完成

## 使用说明

### 环境配置
```bash
# 克隆仓库
git clone https://github.com/your-username/CFLP.git
cd CFLP

# 安装依赖
pip install -r requirements.txt
```

### 运行标准机器学习实验
```bash
# 运行标准实验
python src/main.py
```

### 运行联邦学习实验
```bash
# 使用Docker Compose运行联邦学习实验
cd docker_federated/docker
docker-compose up --build

# 查看训练日志
docker logs -f fl-server
docker logs -f fl-client-1
```

### 准备联邦学习环境
```bash
# 生成gRPC代码
cd docker_federated
python scripts/generate_grpc.py

# 创建客户端数据
python src/data_process/generate_credit_card_data.py
python -c "from src.data_process.generate_credit_card_data import split_data_for_federation; split_data_for_federation(num_clients=3)"
```

## 配置选项

系统提供了多种配置选项以优化模型训练和评估过程:

### 数据处理配置
- `normalize`: 是否对特征进行标准化
- `test_size`: 测试集比例
- `random_seed`: 随机种子

### 模型训练配置
- `param_tuning`: 是否进行参数调优
- `use_smote`: 是否使用SMOTE过采样处理类别不平衡
- `class_weight`: 类别权重设置

### 联邦学习配置
- `n_clients`: 客户端数量
- `n_rounds`: 联邦学习轮数
- `local_epochs`: 本地训练轮次

### Docker联邦学习配置
- 服务端配置
  * `GRPC_SERVER_PORT`: gRPC服务端口
  * `CERT_PATH`: TLS证书路径
  * `REQUIRED_CLIENTS`: 所需客户端数量
  * `MAX_ROUNDS`: 最大训练轮次
- 客户端配置
  * `CLIENT_ID`: 客户端标识
  * `GRPC_SERVER_HOST`: 服务端主机名
  * `GRPC_SERVER_PORT`: 服务端端口
  * `MODEL_TYPE`: 使用的模型类型

## 性能对比

在UCI信用卡违约数据集上的测试结果:

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC-ROC |
|------|--------|--------|--------|--------|---------|
| XGBoost | 0.8256 | 0.6721 | 0.4135 | 0.5128 | 0.7842 |
| 随机森林 | 0.8193 | 0.6458 | 0.3964 | 0.4913 | 0.7731 |
| 逻辑回归 | 0.7854 | 0.5237 | 0.4682 | 0.4942 | 0.7428 |
| 神经网络 | 0.8012 | 0.5831 | 0.4329 | 0.4967 | 0.7615 |

## 联邦学习与中心化学习性能对比

| 训练模式 | 准确率 | 精确率 | 召回率 | F1分数 | AUC-ROC |
|----------|--------|--------|--------|--------|---------|
| 中心化逻辑回归 | 0.7854 | 0.5237 | 0.4682 | 0.4942 | 0.7428 |
| 联邦逻辑回归 | 0.7812 | 0.5184 | 0.4571 | 0.4859 | 0.7356 |
| 中心化神经网络 | 0.8012 | 0.5831 | 0.4329 | 0.4967 | 0.7615 |
| 联邦神经网络 | 0.7986 | 0.5764 | 0.4237 | 0.4886 | 0.7549 |





