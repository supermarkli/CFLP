# CFLP

一个开源的分布式联邦计算平台，支持多个参与方在保护数据隐私的前提下进行联合建模。该平台采用模块化设计，具备高扩展性，支持多种联邦学习算法。



## **功能特性**

- **模块化设计**：
  - 数据加载与预处理：支持稀疏和稠密数据格式。
  - 联邦统计：实现 PSI（Private Set Intersection）、相关性分析等。
  - 特征工程：支持特征选择、归一化和标准化。
  - 联邦学习：支持逻辑回归模型、梯度计算和全局聚合。

- **多节点架构**：
  - **Master Node**：负责模型的全局权重聚合与分发。
  - **Worker Node**：独立计算本地梯度并与 Master Node 交互。

- **gRPC 通信**：
  - 高性能的 gRPC 通信框架实现高效节点间交互。

- **安全与隐私**：
  - 支持 PSI 数据对齐，保护数据隐私。



## **目录结构**

```
CFLP/
├── master_node/                  # 中央服务器代码
│   ├── master_node.py            # Master Node 主程序
│   ├── aggregation.py            # 模型聚合逻辑
│   ├── grpc_server.py            # gRPC 服务端实现
│   ├── requirements.txt          # Master Node 的依赖包
│   ├── Dockerfile                # Master Node Dockerfile
│   └── tests/                    # 测试代码
│       └── test_master_node.py
├── worker_node/                  # 工作节点代码
│   ├── worker_node.py            # Worker Node 主程序
│   ├── local_model.py            # 本地模型训练逻辑
│   ├── grpc_client.py            # gRPC 客户端实现
│   ├── data_processing.py        # 数据加载与预处理模块
│   ├── requirements.txt          # Worker Node 的依赖包
│   ├── Dockerfile                # Worker Node Dockerfile
│   └── tests/                    # 测试代码
│       └── test_worker_node.py
├── proto/                        # gRPC 通信协议定义
│   ├── federated.proto           # gRPC 协议文件
│   ├── federated_pb2.py          # 生成的协议文件（gRPC）
│   ├── federated_pb2_grpc.py     # 生成的服务文件（gRPC）
├── deployment/                   # 部署相关配置
│   ├── docker-compose.yml        # Docker Compose 配置文件
│   ├── kubernetes/               # Kubernetes 配置文件
│   │   ├── master-node.yaml      # Master Node Deployment 配置
│   │   ├── worker-node.yaml      # Worker Node Deployment 配置
│   │   ├── service.yaml          # Kubernetes 服务配置
├── docs/                         # 文档和手册
│   ├── README.md                 # 项目介绍
│   ├── INSTALL.md                # 安装说明
│   └── DEPLOYMENT.md             # 部署说明
├── tests/                        # 全局测试
│   └── test_integration.py       # 集成测试
└── README.md                     # 项目说明
```



## **安装与运行**

### **1. 环境要求**

- Python 版本：3.9+
- Docker：20.10+
- Docker Compose：1.29+

### **2. 安装步骤**

#### **方式 1：使用 Docker Compose**

1. **克隆项目**：
   ```bash
   git clone https://github.com/supermarkli/CFLP.git
   cd cflp
   ```

2. **构建和启动服务**：
   ```bash
   docker-compose up -d
   ```

3. **检查运行状态**：
   ```bash
   docker-compose ps
   ```

#### **方式 2：手动运行**

1. **安装依赖**：
   - Master Node：
     ```bash
     cd master_node
     pip install -r requirements.txt
     ```
   - Worker Node：
     ```bash
     cd worker_node
     pip install -r requirements.txt
     ```

2. **启动 Master Node**：
   ```bash
   python master_node/grpc_server.py

   python -m pytest master_node/tests/test_master_node.py -v
   ```

3. **启动 Worker Node**：
   ```bash
   python worker_node/worker_node.py
   ```



## **用法**

### **1. 添加数据**

将本地数据保存为 CSV 文件，确保包含特征列和目标列。例如：

`local_data.csv`：
```csv
feature1,feature2,target
1.0,2.0,0
1.5,2.5,1
2.0,3.0,0
```

### **2. 配置 Worker Node**

在 `worker_node/worker_node.py` 中指定数据文件路径和目标列名称：
```python
data_path = "local_data.csv"
target_column = "target"
```

### **3. 运行平台**

通过 Docker Compose 或手动启动 Master Node 和 Worker Nodes，完成联邦学习。



## **测试**

### **1. 运行单元测试**

- **Master Node 测试**：
  ```bash
  python -m unittest discover -s master_node -p "test_master_node.py"
  ```

- **Worker Node 测试**：
  ```bash
  python -m unittest discover -s worker_node -p "test_worker_node.py"
  ```

### **2. 运行集成测试**

```bash
python test_integration.py
```



## **扩展与优化**

- **支持更多模型**：
  - 扩展支持线性回归、梯度提升树等模型。

- **隐私保护增强**：
  - 集成同态加密（如 Paillier 加密）实现安全梯度传输。

- **分布式部署**：
  - 在 Kubernetes 集群中部署 Master 和 Worker 节点。

- **监控**：
  - 添加 Prometheus 和 Grafana 实现节点监控和系统性能分析。



## **贡献**

欢迎提交 Issues 和 Pull Requests 来改进本项目！

如果需要进一步修改或扩展内容，请告诉我！



## **当前进展**

### 已完成功能：

1. **基础架构**
   - ✅ Master-Worker架构搭建
   - ✅ gRPC通信框架
   - ✅ 基本的模型训练流程

2. **数据处理**
   - ✅ 基础数据加载和预处理
   - ✅ 支持CSV格式数据

3. **安全性**
   - ✅ Paillier同态加密的基础实现
   - ✅ 简单的PSI实现

4. **部署**
   - ✅ Docker容器化支持
   - ✅ 基础的K8s配置

## **优化路线图**

### 1. 模型扩展
- 添加更多机器学习模型支持
- 实现深度学习模型支持
- 添加模型评估和验证机制

### 2. 安全性增强
- 完善同态加密实现，支持更多加密方案
- 添加差分隐私机制
- 实现安全多方计算（MPC）
- 增加节点认证和授权机制

### 3. 系统性能
- 优化通信效率，实现梯度压缩
- 添加异步训练支持
- 实现模型并行训练
- 添加容错机制和断点续训

### 4. 监控与可视化
- 添加训练过程监控
- 实现Web界面展示训练状态
- 添加性能指标监控（CPU、内存、网络等）
- 集成Prometheus和Grafana

### 5. 数据处理增强
- 支持更多数据格式（如Parquet、HDF5等）
- 添加特征工程工具
- 实现数据质量检查
- 支持增量学习

### 6. 部署优化
- 完善Kubernetes部署配置
- 添加自动扩缩容支持
- 实现多集群部署方案
- 添加服务网格支持

### 7. 测试与文档
- 增加单元测试覆盖率
- 添加性能测试
- 完善API文档
- 添加使用示例和教程

### 8. 工程实践
- 添加CI/CD流程
- 实现版本控制和模型管理
- 添加日志聚合和分析
- 实现配置中心

## **优化阶段规划**

### 第一阶段（核心功能完善）
1. 完善模型支持（特别是深度学习模型）
2. 增强安全性（差分隐私和更完善的加密方案）
3. 添加基础监控系统

### 第二阶段（性能优化）
1. 实现梯度压缩和异步训练
2. 优化通信效率
3. 添加容错机制

### 第三阶段（工程化）
1. 完善部署方案
2. 添加完整的监控系统
3. 实现Web管理界面

### 第四阶段（生产就绪）
1. 完善测试覆盖
2. 添加完整文档
3. 实现DevOps流程