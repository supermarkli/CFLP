import yaml
import pandas as pd
from utils.logging_config import get_logger
from utils.metrics import ModelMetrics

logger = get_logger()

class BaseExperiment:
    """基础实验类,包含所有实验共用的功能"""
    
    def __init__(self, config_path):

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
        self.models[name] = model
        
    def load_data(self):
        """加载数据,跳过第一行"""
        try:
            df = pd.read_csv(self.config['data_path'], skiprows=1)
            logger.info(f"成功加载数据,形状: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise
            
    def save_model(self, model, path):

        try:
            model.save_model(path)
            logger.info(f"模型已保存到: {path}")
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            raise
            
    def load_model(self, model_class, path):
        try:
            model = model_class.load_model(path)
            logger.info(f"成功从{path}加载模型")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise 