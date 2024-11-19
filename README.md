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