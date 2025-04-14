import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
from src.utils.metrics import ModelMetrics
from src.utils.logging_config import get_logger


logger = get_logger()

class FederatedServer:

    def __init__(self):
        self.clients = []
        self.global_model = None
        self.metrics = ModelMetrics()
        self.best_metrics = None
        
    def add_client(self, model):
        self.clients.append(model)
        if self.global_model is None:
            self.global_model = model.get_parameters()
        
    def aggregate_parameters(self, client_parameters):
        if not client_parameters:
            return self.global_model
        
        aggregated_params = {}
        num_clients = len(client_parameters)
        
        if client_parameters:
            first_client_params = client_parameters[0]
            for key in first_client_params.keys():
                param_sum = np.sum([params[key] for params in client_parameters if key in params], axis=0)
                aggregated_params[key] = param_sum / num_clients

        self.global_model = aggregated_params
        return aggregated_params
        
    def train_round(self, round_idx):
        logger.info(f"Federated Learning Round {round_idx + 1}")
        
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
            client.set_parameters(global_parameters)
            
        return client_metrics 

    def get_global_model(self):
        return self.global_model

    def evaluate_global_model(self, test_data):
        if self.global_model is None or not test_data:
            print("Global model not available or no test data provided.")
            return None
        
        if not self.clients:
             print("No clients registered to perform evaluation.")
             return None

        eval_model = self.clients[0]
        eval_model.set_parameters(self.global_model)
        
        metrics = eval_model.evaluate(test_data['X'], test_data['y'])
        self.metrics.update(metrics)
        print(f"Global model evaluation metrics: {metrics}")
        return metrics 