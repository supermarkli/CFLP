from experiments.base import BaseExperiment
from utils.logging_config import get_logger
from federation.fed_client import FederatedClient
from federation.fed_server import FederatedServer
from tqdm import tqdm
import numpy as np

logger = get_logger()

class FederatedExperiment(BaseExperiment):
    """联邦学习实验类"""
    
    def __init__(self, config_path):
        super().__init__(config_path)
        self.fed_server = None
        
    def _split_data_for_clients(self, X, y, n_clients):
        """将数据分割给不同客户端"""
        try:
            # 随机打乱数据
            indices = np.random.permutation(len(X))
            # 计算每个客户端的数据量
            split_size = len(X) // n_clients
            # 分割数据
            data_splits = []
            for i in range(n_clients):
                start_idx = i * split_size
                end_idx = start_idx + split_size if i < n_clients - 1 else len(X)
                client_indices = indices[start_idx:end_idx]
                data_splits.append({
                    'X': X[client_indices],
                    'y': y[client_indices]
                })
            return data_splits
        except Exception as e:
            logger.error(f"数据分割失败: {str(e)}")
            raise
            
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
            
    def train_federated(self, X_val, y_val, n_rounds):
        """训练联邦学习模型"""
        try:
            metrics_history = []
            
            # 使用tqdm显示训练轮次进度
            for round_idx in tqdm(range(n_rounds), desc="联邦学习训练"):
                # 每轮训练和评估
                _ = self.fed_server.train_round(
                    round_idx=round_idx,
                    X_val=X_val,
                    y_val=y_val
                )
                
                # 使用验证集评估全局模型
                global_model = list(self.fed_server.clients.values())[0].model
                y_pred = global_model.predict(X_val)
                y_pred_proba = global_model.predict_proba(X_val)
                
                metrics = self.metrics.calculate_metrics(
                    y_val, y_pred, y_pred_proba
                )
                metrics_history.append(metrics)
                
                # 记录每轮的主要指标
                logger.info(f"轮次 {round_idx + 1}/{n_rounds}:")
                for metric_name, value in metrics.items():
                    logger.info(f"{metric_name}: {value:.4f}")
                    
            return metrics_history
            
        except Exception as e:
            logger.error(f"联邦学习训练失败: {str(e)}")
            raise
            
    def run(self, n_clients=3, n_rounds=10):
        """运行联邦学习实验"""
        try:
            logger.info("开始联邦学习实验...")
            df = self.load_data()
            federated_results = {}
            
            for name, model_template in self.models.items():
                logger.info(f"\n=== 开始 {name} 的联邦学习 ===")
                
                # 数据预处理
                X_train, X_val, X_test, y_train, y_val, y_test = model_template.preprocess_data(df)
                
                # 分割训练数据给客户端
                data_splits = self._split_data_for_clients(X_train, y_train, n_clients)
                
                # 设置客户端
                self.setup_clients(model_template, data_splits)
                
                # 训练联邦学习模型
                metrics_history = self.train_federated(X_val, y_val, n_rounds)
                
                # 最终在测试集上评估
                logger.info(f"\n{name} 在测试集上的最终评估:")
                global_model = list(self.fed_server.clients.values())[0].model
                y_pred = global_model.predict(X_test)
                y_pred_proba = global_model.predict_proba(X_test)
                
                test_metrics = self.metrics.calculate_metrics(
                    y_test, y_pred, y_pred_proba
                )
                
                # 存储结果
                federated_results[name] = {
                    'round_metrics': metrics_history,
                    'final_metrics': test_metrics,
                    'global_model': global_model
                }
                
                # 输出最终测试结果
                for metric_name, value in test_metrics.items():
                    logger.info(f"{metric_name}: {value:.4f}")
                    
                # 保存模型
                if self.config.get('save_models', False):
                    model_path = f"{self.config['model_save_path']}/{name}_federated.model"
                    self.save_model(global_model, model_path)
                    
            return federated_results
            
        except Exception as e:
            logger.error(f"联邦学习实验失败: {str(e)}")
            raise 