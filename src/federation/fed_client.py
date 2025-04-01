from utils.metrics import ModelMetrics
from utils.logging_config import get_logger

logger = get_logger()

class FederatedClient:
    def __init__(self, client_id, model, data=None):
        """初始化联邦学习客户端"""
        self.client_id = client_id
        self.model = model
        self.data = data
        self.metrics = ModelMetrics()
        
    def train(self, epochs):
        """本地训练模型"""
        if self.data is None:
            raise ValueError("No data available for training")
            
        X_train, y_train = self.data
        self.model.train_model(X_train, y_train)
        
        # 计算并记录本地训练指标
        y_pred = self.model.predict(X_train)
        y_pred_proba = self.model.predict_proba(X_train)
        metrics = self.metrics.calculate_metrics(y_train, y_pred, y_pred_proba)
        
        return metrics
        
    def get_parameters(self):
        """获取模型参数"""
        return self.model.get_parameters()
        
    def set_parameters(self, parameters):
        """设置模型参数"""
        self.model.set_parameters(parameters) 