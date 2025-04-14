import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import pandas as pd
from src.utils.logging_config import get_logger
from src.utils.metrics import ModelMetrics

logger = get_logger()

class BaseExperiment:
    """基础实验类,包含所有实验共用的功能"""
    
    def __init__(self):
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models = {}
        self.metrics = ModelMetrics()
            
    def add_model(self, name, model):
        self.models[name] = model
        
    def load_data(self):
        """加载数据"""
        try:
            data_dir = os.path.join(self.PROJECT_ROOT, 'data', 'credit_card')
        
            train_norm_path = os.path.join(data_dir, 'credit_card_train_normalized.csv')
            test_norm_path = os.path.join(data_dir, 'credit_card_test_normalized.csv')
            
            train_raw_path = os.path.join(data_dir, 'credit_card_train_raw.csv')
            test_raw_path = os.path.join(data_dir, 'credit_card_test_raw.csv')
            

            train_norm_df = pd.read_csv(train_norm_path)
            test_norm_df = pd.read_csv(test_norm_path)
            
            # 分离特征和目标变量
            X_train_norm = train_norm_df.drop('target', axis=1)
            y_train_norm = train_norm_df['target']
            X_test_norm = test_norm_df.drop('target', axis=1)
            y_test_norm = test_norm_df['target']
            
            logger.info(f"成功加载标准化数据: 训练集 {X_train_norm.shape}, 测试集 {X_test_norm.shape}")
        
            train_raw_df = pd.read_csv(train_raw_path)
            test_raw_df = pd.read_csv(test_raw_path)
            
            X_train_raw = train_raw_df.drop('target', axis=1)
            y_train_raw = train_raw_df['target']
            X_test_raw = test_raw_df.drop('target', axis=1)
            y_test_raw = test_raw_df['target']
            
            logger.info(f"成功加载非标准化数据: 训练集 {X_train_raw.shape}, 测试集 {X_test_raw.shape}")
            
            if X_train_norm is None and X_train_raw is None:
                raise FileNotFoundError("找不到处理好的数据文件，请先运行数据生成脚本")
                
            return X_train_norm, X_test_norm, y_train_norm, y_test_norm, X_train_raw, X_test_raw, y_train_raw, y_test_raw
            
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