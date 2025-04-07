from experiments.base_experiment import BaseExperiment
from utils.logging_config import get_logger
from federation.fed_client import FederatedClient
from federation.fed_server import FederatedServer
from tqdm import tqdm
import numpy as np
import pandas as pd

logger = get_logger()

class FederatedExperiment(BaseExperiment):
    """联邦学习实验类"""
    
    def __init__(self, config_path):
        super().__init__(config_path)
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
                client_model = type(model_template)(config=self.config)
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
            
    def run(self, n_clients=3, n_rounds=10):
        """运行联邦学习实验"""
        try:
            logger.info("开始联邦学习实验...")
            df = self.load_data()
            federated_results = {}
            
            for name, model_template in self.models.items():
                logger.info(f"\n=== 开始 {name} 的联邦学习 ===")
                
                # 数据预处理,只分训练集和测试集
                X_train, X_test, y_train, y_test = model_template.preprocess_data(df)
                
                # 分割训练数据给客户端
                data_splits = self._split_data_for_clients(X_train, y_train, n_clients)
                
                # 设置客户端
                self.setup_clients(model_template, data_splits)
                
                # 训练联邦学习模型
                client_metrics_history = self.train_federated(n_rounds)
                
                # 最终在测试集上评估
                logger.info(f"\n{name} 在测试集上的最终评估:")
                global_model = list(self.fed_server.clients.values())[0].model
                test_metrics = global_model.evaluate_model(X_test, y_test)

                
                # 存储结果
                federated_results[name] = {
                    'client_metrics_history': client_metrics_history,
                    'final_metrics': test_metrics,
                    'global_model': global_model
                }
                
                    
                # 保存模型
                if self.config.get('save_models', False):
                    model_path = f"{self.config['model_save_path']}/{name}_federated.model"
                    self.save_model(global_model, model_path)
            
            # 比较所有模型的性能
            self.compare_models(federated_results)
                    
            return federated_results
            
        except Exception as e:
            logger.error(f"联邦学习实验失败: {str(e)}")
            raise 