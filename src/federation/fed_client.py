from utils.metrics import ModelMetrics
from utils.logging_config import get_logger
from sklearn.model_selection import train_test_split

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
        
        # 在初始化时划分训练集和测试集
        if data is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'], test_size=0.2, random_state=42
            )
            self.train_data = {'X': X_train, 'y': y_train}
            self.test_data = {'X': X_test, 'y': y_test}
        else:
            self.train_data = None
            self.test_data = None
        
    def train(self, epochs):
        """本地训练模型"""
        if self.train_data is None:
            raise ValueError("No data available for training")
        
        # 根据模型类型设置训练轮数
        if hasattr(self.model, 'epochs'):  # 神经网络模型
            self.model.epochs = epochs
        elif hasattr(self.model, 'max_iter'):  # 逻辑回归模型
            self.model.max_iter = epochs
        elif hasattr(self.model, 'num_boost_round'):  # XGBoost模型
            self.model.num_boost_round = epochs
        elif hasattr(self.model, 'n_estimators'):  # 随机森林模型
            self.model.n_estimators = epochs
        
        # 使用训练集训练模型
        self.model.train_model(self.train_data['X'], self.train_data['y'])
        
        # 使用测试集评估模型
        metrics = self.model.evaluate_model(self.test_data['X'], self.test_data['y'])
        
        return metrics
        
    def get_parameters(self):
        """获取模型参数"""
        return self.model.get_parameters()
        
    def set_parameters(self, parameters):
        """设置模型参数"""
        self.model.set_parameters(parameters) 