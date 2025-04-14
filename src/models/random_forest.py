import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

from src.models.base_model import BaseModel
from src.utils.metrics import ModelMetrics
from src.utils.logging_config import get_logger

logger = get_logger()

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.param_tuning = False
        self.model = RandomForestClassifier()
        self.name = "RandomForest"
        self.normalize = False
        self.metrics = ModelMetrics()
        self.best_threshold = 0.5
        
    def grid_search_cv(self, X, y, param_grid):
        logger.info("\n=== 开始网格搜索 ===")
        
        best_score = 0
        best_params = None
        best_model = None
        
        # 进行网格搜索
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for min_samples_split in param_grid['min_samples_split']:
                    for min_samples_leaf in param_grid['min_samples_leaf']:
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'random_state': 42,
                            'n_jobs': -1
                        }
                        
                        # 使用当前参数创建模型
                        rf = RandomForestClassifier(**params)
                        
                        # 使用交叉验证评估参数
                        cv_scores = cross_val_score(
                            rf, X, y, 
                            cv=5, 
                            scoring='roc_auc'
                        )
                        mean_score = cv_scores.mean()
                        
                        # 输出当前参数组合的结果
                        logger.info(f"参数: {params}")
                        logger.info(f"交叉验证AUC分数: {mean_score:.4f}")
                        
                        # 更新最优参数和模型
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = params
                            # 使用最优参数重新训练模型
                            best_model = RandomForestClassifier(**params)
                            best_model.fit(X, y)
        
        logger.info(f"网格搜索完成")
        logger.info(f"最优参数: {best_params}")
        logger.info(f"最优交叉验证AUC分数: {best_score:.4f}")
        
        return best_model, best_params

    def train_model(self, X_train, y_train):
        logger.info(f"模型配置:")
        logger.info(f"- 参数调优: {self.param_tuning}")
        logger.info(f"- 训练数据形状: {X_train.shape}")
        logger.info(f"训练集类别分布:")
        logger.info(f"- 正样本数量: {np.sum(y_train == 1)}")
        logger.info(f"- 负样本数量: {np.sum(y_train == 0)}")
        
        if self.param_tuning:
            # 定义参数网格
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # 执行网格搜索
            self.model, best_params = self.grid_search_cv(X_train, y_train, param_grid)
        else:
            # 使用默认参数
            self.model = RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        # 交叉验证评估
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=5, scoring='roc_auc'
        )
        logger.info("交叉验证结果:")
        logger.info(f"- 平均AUC: {cv_scores.mean():.4f}")
        logger.info(f"- 标准差: {cv_scores.std() * 2:.4f}")

    def predict(self, X_test):
        y_pred_proba = self.predict_proba(X_test)
        return (y_pred_proba > self.best_threshold).astype(int)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]  # 取正类概率

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
        
        # logger.info(f"找到最优阈值: {best_threshold:.2f} (准确率: {best_accuracy:.4f})")
        return best_threshold

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise ValueError("错误：模型未训练")
        
        logger.info("\n=== 模型评估 ===")
        logger.info(f"测试数据形状: {X_test.shape}")
        
        # 获取预测结果
        y_pred_proba = self.predict_proba(X_test)
        
        # 找到最优阈值
        self.best_threshold = self.find_best_threshold(y_test, y_pred_proba)
        logger.info(f"使用阈值: {self.best_threshold:.4f}")
        
        # 使用最优阈值进行预测
        y_pred = (y_pred_proba > self.best_threshold).astype(int)
        
        # 计算评估指标
        test_metrics = self.metrics.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return test_metrics

    def get_parameters(self):
        """获取模型参数"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        return {
            'estimators': self.model.estimators_,
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_,
            'n_outputs': self.model.n_outputs_,
            'feature_importances': self.model.feature_importances_
        }

    def set_parameters(self, parameters):
        """设置模型参数"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        self.model.estimators_ = parameters['estimators']
        self.model.n_features_in_ = parameters['n_features']
        self.model.n_classes_ = parameters['n_classes']
        self.model.n_outputs_ = parameters['n_outputs']
        self.model.feature_importances_ = parameters['feature_importances']
