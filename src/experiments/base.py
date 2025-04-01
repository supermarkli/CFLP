import yaml
import pandas as pd
from utils.logging_config import get_logger
from utils.metrics import ModelMetrics

logger = get_logger()

class BaseExperiment:
    """基础实验类,包含所有实验共用的功能"""
    
    def __init__(self, config_path):
        """
        初始化实验
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.metrics = ModelMetrics()
        
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise
            
    def add_model(self, name, model):
        """
        添加模型到实验中
        
        Args:
            name: 模型名称
            model: 模型实例
        """
        self.models[name] = model
        
    def load_data(self):
        """加载数据"""
        try:
            df = pd.read_csv(self.config['data_path'])
            logger.info(f"成功加载数据,形状: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise
            
    def save_model(self, model, path):
        """
        保存模型
        
        Args:
            model: 要保存的模型实例
            path: 保存路径
        """
        try:
            model.save_model(path)
            logger.info(f"模型已保存到: {path}")
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            raise
            
    def load_model(self, model_class, path):
        """
        加载模型
        
        Args:
            model_class: 模型类
            path: 模型文件路径
        Returns:
            加载的模型实例
        """
        try:
            model = model_class.load_model(path)
            logger.info(f"成功从{path}加载模型")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise 