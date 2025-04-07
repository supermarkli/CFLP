from utils.metrics import ModelMetrics
from utils.logging_config import get_logger

logger = get_logger()

class FederatedServer:

    def __init__(self):
        self.clients = {}
        self.global_model = None
        self.metrics = ModelMetrics()
        self.best_metrics = None
        
    def add_client(self, client):
        self.clients[client.client_id] = client
        
    def aggregate_parameters(self, client_parameters):
        n_clients = len(client_parameters)
        aggregated = {}

        for param_name in client_parameters[0].keys():
            aggregated[param_name] = sum(
                params[param_name] for params in client_parameters
            ) / n_clients
            
        return aggregated
        
    def train_round(self, round_idx):
        logger.info(f"Federated Learning Round {round_idx + 1}")
        
        # 1. 在每个客户端上进行本地训练
        client_metrics = {}
        for client_id, client in self.clients.items():
            metrics = client.train(epochs=10)
            client_metrics[client_id] = metrics
            
        # 2. 收集并聚合所有客户端的参数
        client_parameters = [
            client.get_parameters() 
            for client in self.clients.values()
        ]
        global_parameters = self.aggregate_parameters(client_parameters)
        
        # 3. 将聚合后的参数更新到所有客户端
        for client in self.clients.values():
            client.set_parameters(global_parameters)
            
        return client_metrics 