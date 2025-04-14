import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
from src.utils.metrics import ModelMetrics
from src.utils.logging_config import get_logger


logger = get_logger()

class FederatedServer:

    def __init__(self, test_data=None):
        self.clients = []
        self.global_model = None
        self.metrics = ModelMetrics()
        self.best_metrics = None
        self.test_data = test_data
        
    def add_client(self, client):
        """添加联邦学习客户端
        
        Args:
            client: FederatedClient实例
        """
        self.clients.append(client)
        if self.global_model is None:
            self.global_model = client.model
        
    def aggregate_parameters(self, client_parameters):
        if not client_parameters:
            logger.warning("客户端参数为空,将返回当前全局模型")
            return self.global_model
        
        aggregated_params = {}
        num_clients = len(client_parameters)
        
        if client_parameters:
            first_client_params = client_parameters[0]
            for key in first_client_params.keys():
                param_sum = np.sum([params[key] for params in client_parameters if key in params], axis=0)
                aggregated_params[key] = param_sum / num_clients

        return aggregated_params
        
    def train_round(self, round_idx, total_rounds=None):
        logger.info(f"\n=== {self.global_model.name} 联邦学习第 {round_idx + 1} 轮 ===")
        
        # 1. 在每个客户端上进行本地训练
        client_metrics = {}
        for client in self.clients:
            metrics = client.train(epochs=10)
            client_metrics[client] = metrics
            
        # 2. 收集并聚合所有客户端的参数
        client_parameters = [client.get_parameters() for client in self.clients]
        global_parameters = self.aggregate_parameters(client_parameters)
        
        # 3. 将聚合后的参数更新到所有客户端
        for client in self.clients:
            client.model.set_parameters(global_parameters)
        
        # 4. 评估全局模型
        self.global_model = self.clients[0].model
        metrics = self.global_model.evaluate_model(self.test_data['X'], self.test_data['y'])
        
        # 根据是否是最后一轮决定日志前缀
        prefix = "最终评估 - " if total_rounds and round_idx == total_rounds - 1 else ""
        logger.info(f"{prefix}全局模型评估指标:")
        logger.info(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"{prefix}Precision: {metrics['precision']:.4f}")
        logger.info(f"{prefix}Recall: {metrics['recall']:.4f}")
        logger.info(f"{prefix}F1 Score: {metrics['f1']:.4f}")
        logger.info(f"{prefix}AUC-ROC: {metrics['auc_roc']:.4f}")

        return client_metrics

    def get_global_model(self):
        return self.global_model

