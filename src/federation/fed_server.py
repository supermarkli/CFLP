from utils.metrics import ModelMetrics
from utils.logging_config import get_logger

logger = get_logger()

class FederatedServer:
    """联邦学习服务器类
    
    负责协调多个客户端的联邦学习过程,包括:
    1. 管理客户端加入和退出
    2. 聚合客户端模型参数
    3. 协调训练过程
    4. 评估全局模型性能
    
    属性:
        clients (dict): 存储所有客户端,键为客户端ID,值为客户端实例
        global_model: 全局模型实例
        metrics: 用于计算评估指标的工具类
        best_metrics: 记录训练过程中的最佳指标
    """
    def __init__(self):
        """初始化联邦学习服务器
        
        初始化必要的属性,包括:
        - 客户端字典
        - 全局模型
        - 评估指标计算器
        - 最佳指标记录
        """
        self.clients = {}
        self.global_model = None
        self.metrics = ModelMetrics()
        self.best_metrics = None
        
    def add_client(self, client):
        """添加联邦学习客户端
        
        Args:
            client: 要添加的客户端实例
            
        功能:
            将新的客户端添加到服务器的客户端字典中
        """
        self.clients[client.client_id] = client
        
    def aggregate_parameters(self, client_parameters):
        """聚合客户端参数(FedAvg算法)
        
        Args:
            client_parameters (list): 所有客户端的模型参数列表
            
        Returns:
            dict: 聚合后的全局模型参数
            
        实现细节:
        1. 获取参与聚合的客户端数量
        2. 对每个参数进行简单平均
        3. 返回聚合后的参数字典
        
        注意:
        - 使用FedAvg算法,即简单平均策略
        - 要求所有客户端模型结构相同
        """
        # 获取客户端数量
        n_clients = len(client_parameters)
        aggregated = {}
        
        # 对每个参数进行平均聚合
        for param_name in client_parameters[0].keys():
            aggregated[param_name] = sum(
                params[param_name] for params in client_parameters
            ) / n_clients
            
        return aggregated
        
    def train_round(self, round_idx, X_val=None, y_val=None):
        """执行一轮联邦学习训练
        
        Args:
            round_idx (int): 当前训练轮次索引
            X_val: 用于评估的验证集特征
            y_val: 用于评估的验证集标签
            
        Returns:
            dict: 包含所有客户端的训练指标
            
        执行流程:
        1. 客户端本地训练:
           - 每个客户端使用本地数据训练模型
           - 记录每个客户端的训练指标
           
        2. 参数聚合:
           - 收集所有客户端的模型参数
           - 使用FedAvg算法聚合参数
           
        3. 参数更新:
           - 将聚合后的参数分发给所有客户端
           
        4. 模型评估:
           - 使用验证集评估全局模型性能
           - 记录并更新最佳指标
           
        注意:
        - 每轮训练后所有客户端具有相同的模型参数
        - 使用F1分数作为模型选择的指标
        - 通过日志记录训练过程中的关键信息
        """
        # 输出当前训练轮次
        logger.info(f"\nFederated Learning Round {round_idx + 1}")
        
        # 1. 在每个客户端上进行本地训练
        client_metrics = {}
        for client_id, client in self.clients.items():
            metrics = client.train(epochs=1)
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
            
        # 4. 在验证集上评估全局模型
        if X_val is not None and y_val is not None:
            # 由于所有客户端参数相同,使用第一个客户端的模型进行评估
            model = list(self.clients.values())[0].model
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
            
            # 计算评估指标
            metrics = self.metrics.calculate_metrics(y_val, y_pred, y_pred_proba)
            logger.info("Global Model Metrics:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f} | "
                       f"F1: {metrics['f1']:.4f} | "
                       f"AUC-ROC: {metrics['auc_roc']:.4f}")
            
            # 更新最佳指标(如果当前模型更好)
            if self.best_metrics is None or metrics['f1'] > self.best_metrics['f1']:
                self.best_metrics = metrics
                
        return client_metrics 