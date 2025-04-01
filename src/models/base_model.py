import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
from data_process.credit_card import CreditCardDataPreprocessor
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def predict_proba(self, X):
        """返回预测概率
        
        Returns:
            numpy array: 预测的正类概率(一维数组)
        """
        pass

class BaseModel:
    def __init__(self, config):
        """初始化基础模型类
        
        参数:
            config: 配置字典,包含模型参数等配置信息。可以包含:
                - 数据相关配置(数据路径、划分比例等)
                - 模型相关配置(学习率、迭代次数等)
                - 训练相关配置(批次大小、验证比例等)
        """
        self.config = config
        self.preprocessor = None
        self.model = None
        self.feature_names = None
        self.normalize = None
        
    def preprocess_data(self, df):
        """对输入数据进行预处理
        
        对数据进行以下处理:
        1. 特征和目标变量分离
        2. 训练集、验证集、测试集划分
        3. 特征工程(标准化、独热编码等)
        
        参数:
            df: pandas DataFrame, 原始输入数据
            
        返回:
            X_train: 训练集特征
            X_val: 验证集特征
            X_test: 测试集特征
            y_train: 训练集标签
            y_val: 验证集标签
            y_test: 测试集标签
        """
        # 先保存特征名称
        X, y = self.preprocessor.split_features_target(df)
        self.feature_names = X.columns.tolist()
        
        # 数据清理 
        X = self.preprocessor.clean_data(X)
        
        # 进行数据预处理(划分数据集和特征工程)
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.fit_transform(X, y)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def train_model(self, X_train, y_train):
        """训练模型
        
        使用训练数据拟合模型。具体实现由子类完成。
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
        """
        raise NotImplementedError("Subclass must implement train_model method")
        
    def evaluate_model(self, X_test, y_test):
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
            dict: 包含各项评估指标的字典
        """
        raise NotImplementedError("Subclass must implement evaluate_model method")
        
    def analyze_feature_importance(self):
        """分析特征重要性
        
        计算并展示各个特征的重要性分数。
        可以包括:
        - 特征重要性排序
        - 重要性分数可视化
        - 重要特征分析报告
        """
        raise NotImplementedError("Subclass must implement analyze_feature_importance method")
        
    def get_parameters(self):
        """获取模型参数
        
        返回模型的当前参数设置。
        用于模型保存和迁移。
        
        返回:
            dict: 包含模型参数的字典
        """
        raise NotImplementedError("Subclass must implement get_parameters method")
        
    def set_parameters(self, parameters):
        """设置模型参数
        
        使用给定的参数更新模型。
        用于模型加载和参数更新。
        
        参数:
            parameters: dict, 要设置的模型参数
        """
        raise NotImplementedError("Subclass must implement set_parameters method") 
        
