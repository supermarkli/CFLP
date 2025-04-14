import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from src.utils.metrics import ModelMetrics
from src.utils.logging_config import get_logger

logger = get_logger()

class FederatedClient:
    def __init__(self, client_id, model, data=None):
        """初始化联邦学习客户端
        
        Args:
            client_id: 客户端ID
            model: 本地模型
            data: 包含特征X和标签y的数据字典
        """
        self.client_id = client_id
        self.model = model
        self.metrics = ModelMetrics()
        
        if data is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'], test_size=0.2, random_state=42
            )
            self.train_data = {'X': X_train, 'y': y_train}
            self.test_data = {'X': X_test, 'y': y_test}
        else:
            self.train_data = None
            self.test_data = None
        
    def train(self, epochs=10):
        """本地训练模型"""
        if self.train_data is None:
            logger.warning(f"Client {self.client_id}: No training data available.")
            return self.metrics.get_metrics()

        if hasattr(self.model, 'epochs'):  # 神经网络模型
            self.model.epochs = epochs
        elif hasattr(self.model, 'max_iter'):  # 逻辑回归模型
            self.model.max_iter = epochs

        self.model.train_model(self.train_data['X'], self.train_data['y'])
        metrics = self.model.evaluate_model(self.test_data['X'], self.test_data['y'])
        return metrics
        
    def get_parameters(self):
        """获取模型参数"""
        return self.model.get_parameters()
        
    def set_parameters(self, parameters):
        """设置模型参数"""
        self.model.set_parameters(parameters)
        logger.info(f"Client {self.client_id}: Model parameters updated.")
        