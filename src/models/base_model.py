import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
from abc import ABC, abstractmethod

class BaseModel(ABC):

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.normalize: Optional[bool] = None
        
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
        
