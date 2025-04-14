import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.experiments.base_experiment import BaseExperiment
from src.federation.fed_client import FederatedClient
from src.federation.fed_server import FederatedServer
from src.utils.logging_config import get_logger

logger = get_logger()

class FederatedExperiment(BaseExperiment):
    """联邦学习实验类"""
    
    def __init__(self):
        super().__init__()
        self.fed_server = None
        
    def _split_data_for_clients(self, X, y, n_clients):
        try:
            
            # 确保X和y是numpy数组
            X = np.asarray(X)
            y = np.asarray(y)
            
            # 随机打乱数据
            logger.info(f"开始数据分割,总样本数: {len(X)}, 客户端数: {n_clients}")
            indices = np.random.permutation(len(X))
            
            # 计算每个客户端的数据量
            split_size = len(X) // n_clients
            logger.info(f"每个客户端平均数据量: {split_size}")
            
            # 分割数据
            data_splits = []
            for i in range(n_clients):
                start_idx = i * split_size
                end_idx = start_idx + split_size if i < n_clients - 1 else len(X)
                client_indices = indices[start_idx:end_idx]
                
                client_data = {
                    'X': X[client_indices],
                    'y': y[client_indices]
                }
                data_splits.append(client_data)
                
                logger.info(f"客户端{i}数据量: {len(client_indices)}")
                
            return data_splits
            
        except ValueError as e:
            logger.error(f"数据分割参数错误: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"数据分割失败: {str(e)}")
            raise RuntimeError(f"数据分割过程中发生错误: {str(e)}")
            
    def setup_clients(self, model_template, data_splits):
        """设置联邦学习客户端"""
        try:
            self.fed_server = FederatedServer()
            
            for i, data in enumerate(data_splits):
                # 为每个客户端创建独立的模型实例
                client_model = type(model_template)()
                client = FederatedClient(
                    client_id=f"client_{i}",
                    model=client_model,
                    data=data
                )
                self.fed_server.add_client(client)
                
        except Exception as e:
            logger.error(f"设置客户端失败: {str(e)}")
            raise
            
    def train_federated(self, n_rounds):
        """训练联邦学习模型"""
        try:
            client_metrics_history = []
            
            # 使用tqdm显示训练轮次进度
            for round_idx in tqdm(range(n_rounds), desc="联邦学习训练"):
                # 每轮训练
                round_metrics = self.fed_server.train_round(round_idx=round_idx)
                client_metrics_history.append(round_metrics)
                
                # 记录每轮的进度
                logger.info(f"轮次 {round_idx + 1}/{n_rounds} 完成")
                    
            return client_metrics_history
            
        except Exception as e:
            logger.error(f"联邦学习训练失败: {str(e)}")
            raise
            
    def compare_models(self, federated_results):
        """比较所有模型在联邦学习中的性能
        
        Args:
            federated_results: 包含所有模型训练结果的字典
            
        功能:
        1. 显示所有模型在测试集上的最终性能对比
        2. 找出性能最好的模型
        3. 输出详细的评估指标
        """
        try:
            # 提取每个模型的最终指标
            final_metrics = {}
            for name, results in federated_results.items():
                final_metrics[name] = results['final_metrics']
            
            # 转换为DataFrame进行比较
            results_df = pd.DataFrame(final_metrics).T
            
            # 显示模型对比表格
            logger.info("\n=== 联邦学习模型性能对比 ===")
            logger.info("\n" + results_df.to_string())
                                
            return results_df
            
        except Exception as e:
            logger.error(f"模型对比失败: {str(e)}")
            raise
            

        """加载数据"""
        try:
            # 直接读取处理好的CSV文件
            data_dir = os.path.join(self.PROJECT_ROOT, 'data', 'credit_card')
            
            # 读取标准化数据
            train_norm_path = os.path.join(data_dir, 'credit_card_train_normalized.csv')
            test_norm_path = os.path.join(data_dir, 'credit_card_test_normalized.csv')
            
            # 读取非标准化数据
            train_raw_path = os.path.join(data_dir, 'credit_card_train_raw.csv')
            test_raw_path = os.path.join(data_dir, 'credit_card_test_raw.csv')
            
            # 检查文件是否存在
            if not os.path.exists(train_norm_path) or not os.path.exists(test_norm_path):
                logger.warning("标准化数据文件不存在")
                X_train_norm = None
                X_test_norm = None
                y_train_norm = None
                y_test_norm = None
            else:
                # 读取标准化训练集和测试集
                train_norm_df = pd.read_csv(train_norm_path)
                test_norm_df = pd.read_csv(test_norm_path)
                
                # 分离特征和目标变量
                X_train_norm = train_norm_df.drop('target', axis=1)
                y_train_norm = train_norm_df['target']
                X_test_norm = test_norm_df.drop('target', axis=1)
                y_test_norm = test_norm_df['target']
                
                logger.info(f"成功加载标准化数据: 训练集 {X_train_norm.shape}, 测试集 {X_test_norm.shape}")
            
            if not os.path.exists(train_raw_path) or not os.path.exists(test_raw_path):
                logger.warning("非标准化数据文件不存在")
                X_train_raw = None
                X_test_raw = None
                y_train_raw = None
                y_test_raw = None
            else:
                # 读取非标准化训练集和测试集
                train_raw_df = pd.read_csv(train_raw_path)
                test_raw_df = pd.read_csv(test_raw_path)
                
                # 分离特征和目标变量
                X_train_raw = train_raw_df.drop('target', axis=1)
                y_train_raw = train_raw_df['target']
                X_test_raw = test_raw_df.drop('target', axis=1)
                y_test_raw = test_raw_df['target']
                
                logger.info(f"成功加载非标准化数据: 训练集 {X_train_raw.shape}, 测试集 {X_test_raw.shape}")
            
            # 检查是否至少有一套数据可用
            if X_train_norm is None and X_train_raw is None:
                raise FileNotFoundError("找不到处理好的数据文件，请先运行数据生成脚本")
                
            return X_train_norm, X_test_norm, y_train_norm, y_test_norm, X_train_raw, X_test_raw, y_train_raw, y_test_raw
            
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise
            
    def run(self, n_clients=3, n_rounds=10):
        """运行联邦学习实验"""
        try:
            
            X_train_norm, X_test_norm, y_train_norm, y_test_norm, X_train_raw, X_test_raw, y_train_raw, y_test_raw = self.load_data()
            
            federated_results = {}
            
            for name, model_template in self.models.items():
                logger.info(f"\n=== 开始 {name} 的联邦学习 ===")
                
                if hasattr(model_template, 'normalize') and model_template.normalize:
                    if X_train_norm is None:
                        logger.warning(f"模型 {name} 需要标准化数据，但标准化数据不可用，跳过该模型")
                        continue
                    X_train, X_test, y_train, y_test = X_train_norm, X_test_norm, y_train_norm, y_test_norm
                    logger.info(f"使用标准化数据训练模型 {name}")
                else:
                    if X_train_raw is None:
                        logger.warning(f"模型 {name} 需要非标准化数据，但非标准化数据不可用，跳过该模型")
                        continue
                    X_train, X_test, y_train, y_test = X_train_raw, X_test_raw, y_train_raw, y_test_raw
                    logger.info(f"使用非标准化数据训练模型 {name}")
                
                # 将DataFrame转换为NumPy数组
                if isinstance(X_train, pd.DataFrame):
                    X_train = X_train.to_numpy()
                if isinstance(X_test, pd.DataFrame):
                    X_test = X_test.to_numpy()
                if isinstance(y_train, pd.Series):
                    y_train = y_train.to_numpy()
                if isinstance(y_test, pd.Series):
                    y_test = y_test.to_numpy()
                
                # 分割训练数据给客户端
                data_splits = self._split_data_for_clients(X_train, y_train, n_clients)
                
                # 设置客户端
                self.setup_clients(model_template, data_splits)
                
                # 训练联邦学习模型
                client_metrics_history = self.train_federated(n_rounds)
                
                # 最终在测试集上评估
                logger.info(f"\n{name} 在测试集上的最终评估:")
                global_model = self.fed_server.clients[0].model
                test_metrics = global_model.evaluate_model(X_test, y_test)
                
                # 存储结果
                federated_results[name] = {
                    'client_metrics_history': client_metrics_history,
                    'final_metrics': test_metrics,
                    'global_model': global_model
                }
                
            
            # 比较所有模型的性能
            self.compare_models(federated_results)
                    
            return federated_results
            
        except Exception as e:
            logger.error(f"联邦学习实验失败: {str(e)}")
            raise 