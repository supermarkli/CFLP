from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.logging_config import get_logger

logger = get_logger()

class BaseDataPreprocessor(ABC):
    """Base data preprocessor"""
    def __init__(self, config):
        self.config = config
        self.feature_columns = None
        self.preprocessors = []
                
    @abstractmethod
    def split_features_target(self, df):
        """Split features and target variable"""
        pass
        
    def add_preprocessor(self, preprocessor):
        """Add preprocessing step"""
        self.preprocessors.append(preprocessor)
        
    def fit_transform(self, X, y):
        """通用的预处理流程"""
        try:
            self.feature_columns = X.columns
            
            # 首先划分训练集和测试集
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=self.config.get('data', {}).get('test_size', 0.2),
                random_state=self.config.get('base', {}).get('random_seed', 42)
            )
            
            # 再从临时训练集中划分出验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=self.config.get('data', {}).get('val_size', 0.2),
                random_state=self.config.get('base', {}).get('random_seed', 42)
            )
            
            # 对每个数据集应用预处理
            X_train_processed = X_train.copy()
            X_val_processed = X_val.copy()
            X_test_processed = X_test.copy()
            
            for preprocessor in self.preprocessors:
                X_train_processed = preprocessor.fit_transform(X_train_processed)
                X_val_processed = preprocessor.transform(X_val_processed)
                X_test_processed = preprocessor.transform(X_test_processed)
            
            logger.info(f"Training set size: {len(X_train)}")
            logger.info(f"X_train_processed first 5 rows:\n{pd.DataFrame(X_train_processed)[:5]}")
            logger.info(f"Validation set size: {len(X_val)}")
            logger.info(f"Test set size: {len(X_test)}")
            logger.info("\n=== Data preprocessing completed ===")

            return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise 

