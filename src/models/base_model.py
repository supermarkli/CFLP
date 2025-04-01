import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
from data_process.credit_card import CreditCardDataPreprocessor
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """机器学习模型的基类
    
    实现了基本的模型训练、评估和特征分析功能。
    所有具体的模型实现都应该继承这个基类。
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化基础模型类
        
        参数:
            config: 配置字典,包含模型参数等配置信息。可以包含:
                - 数据相关配置(数据路径、划分比例等)
                - 模型相关配置(学习率、迭代次数等)
                - 训练相关配置(批次大小、验证比例等)
        """
        self.config = config
        self.preprocessor: Optional[CreditCardDataPreprocessor] = None
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.normalize: Optional[bool] = None
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                        pd.Series, pd.Series, pd.Series]:
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be initialized before calling preprocess_data")
            
        # 先保存特征名称
        X, y = self.preprocessor.split_features_target(df)
        self.feature_names = X.columns.tolist()
        
        # 数据清理 
        X = self.preprocessor.clean_data(X)
        
        # 进行数据预处理(划分数据集和特征工程)
        return self.preprocessor.fit_transform(X, y)
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测样本属于正类的概率
        
        参数:
            X: 特征矩阵
            
        返回:
            预测的正类概率(一维数组)
        """
        pass
        
    @abstractmethod
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """训练模型
        
        使用训练数据拟合模型。具体实现由子类完成。
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
        """
        pass
        
    @abstractmethod
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """评估模型性能
        
        计算各项评估指标,包括:
        - 准确率(Accuracy)
        - 精确率(Precision) 
        - 召回率(Recall)
        - F1分数
        - AUC-ROC分数
        
        参数:
            X_test: 测试集特征
            y_test: 测试集标签
            
        返回:
            包含各项评估指标的字典
        """
        pass
        
    @abstractmethod
    def analyze_feature_importance(self) -> Dict[str, float]:
        """分析特征重要性
        
        计算并返回各个特征的重要性分数。
        
        返回:
            特征名称到重要性分数的映射字典
        """
        pass
        
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数
        
        返回:
            包含模型参数的字典
        """
        pass
        
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """设置模型参数
        
        参数:
            parameters: 要设置的模型参数字典
        """
        pass
        
