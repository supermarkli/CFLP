import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE

from src.models.base_model import BaseModel
from src.utils.metrics import ModelMetrics
from src.utils.logging_config import get_logger

logger = get_logger() 

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.param_tuning = False
        self.metrics = ModelMetrics()
        self.name = "LogisticRegression"
        self.normalize = True
        self.best_threshold = 0.5
        self.use_smote = False
        
        self.model = SGDClassifier(
            loss='log_loss',
            eta0=0.1,
            learning_rate='adaptive',
            max_iter=1000,
            class_weight='balanced'
        )
        self.model.coef_ = np.zeros((1, 26))  
        self.model.intercept_ = np.zeros(1)
        
    def train_model(self, X_train, y_train):
        """训练逻辑回归模型"""
        if self.param_tuning:
            param_grid = {
                'eta0': [0.1, 0.01, 0.001],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'max_iter': [1000, 2000],
                'class_weight': ['balanced', None]
            }
            best_params = self.grid_search_cv(X_train, y_train, param_grid)
            self.model = SGDClassifier(loss='log_loss', **best_params)
        
        if self.use_smote:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            self.model.fit(X_train_resampled, y_train_resampled)
        else:
            self.model.fit(X_train, y_train)
            
        
    def grid_search_cv(self, X, y, param_grid):
        """网格搜索寻找最优参数"""
        logger.info("Starting grid search...")
        
        best_score = 0
        best_params = None
        
        # 进行网格搜索
        for eta0 in param_grid['eta0']:
            for learning_rate in param_grid['learning_rate']:
                for max_iter in param_grid['max_iter']:
                    for class_weight in param_grid['class_weight']:
                        params = {
                            'eta0': eta0,
                            'learning_rate': learning_rate,
                            'max_iter': max_iter,
                            'class_weight': class_weight,
                            'random_state': 42
                        }
                        
                        # 使用当前参数创建模型
                        model = SGDClassifier(loss='log_loss', **params)
                        
                        # 使用交叉验证评估参数
                        cv_scores = cross_val_score(
                            model, X, y,
                            cv=5,
                            scoring='roc_auc'
                        )
                        mean_score = cv_scores.mean()
                        
                        # 输出当前参数组合的结果
                        logger.info(f"Parameters: {params}")
                        logger.info(f"Cross-validation score: {mean_score:.4f}")
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = params
        
        logger.info(f"Grid search completed. Best parameters: {best_params}")
        logger.info(f"Best cross-validation score: {best_score:.4f}")
        
        return best_params
        
    def find_best_threshold(self, y_test, y_pred_proba):
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_accuracy = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        # logger.info(f"Found best threshold: {best_threshold:.2f} (Accuracy: {best_accuracy:.4f})")
        return best_threshold

    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("Model not trained")
            
        
        # 获取预测概率
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 找到最优阈值
        self.best_threshold = self.find_best_threshold(y_test, y_pred_proba)
        
        # 使用最优阈值进行预测
        y_pred = (y_pred_proba > self.best_threshold).astype(int)
        
        # 计算评估指标
        test_metrics = self.metrics.calculate_metrics(y_test, y_pred, y_pred_proba)
                
        return test_metrics
        
    def predict(self, X_test):
        y_pred_proba = self.predict_proba(X_test)
        return (y_pred_proba > self.best_threshold).astype(int)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

    def get_parameters(self):
        """获取模型参数"""
        if not hasattr(self.model, 'coef_'):
            # 如果模型还没有参数，返回初始化的零参数
            return {
                'coef': np.zeros((1, 26)),
                'intercept': np.zeros(1)
            }
        return {
            'coef': self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy()
        }
        
    def set_parameters(self, parameters):
        """设置模型参数"""
        if not parameters:
            return
            
        if 'coef' in parameters:
            if not hasattr(self.model, 'coef_'):
                self.model.coef_ = parameters['coef']
            else:
                self.model.coef_ = parameters['coef']
                
        if 'intercept' in parameters:
            if not hasattr(self.model, 'intercept_'):
                self.model.intercept_ = parameters['intercept']
            else:
                self.model.intercept_ = parameters['intercept']
