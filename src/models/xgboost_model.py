import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from src.models.base_model import BaseModel
from src.utils.metrics import ModelMetrics
from src.utils.logging_config import get_logger

logger = get_logger()

class XGBoostModel(BaseModel):
    
    def __init__(self):

        super().__init__()
        self.param_tuning = False
        self.model = None
        self.metrics = ModelMetrics()  # 初始化评估器
        self.name = "XGBoost"  # 添加模型名称属性
        self.best_threshold = 0.5  # 添加最优阈值属性
        self.normalize = False
        
        # 调整参数以提高召回率
        self.params = {
            'max_depth': 4,             # 略微增加深度
            'eta': 0.01,
            'objective': 'binary:logistic',
            'eval_metric': ['auc'],  
            'min_child_weight': 2,      # 降低以允许更多分裂
            'subsample': 0.8,           # 增加采样比例
            'colsample_bytree': 0.7,   # 增加特征采样
        }
        
    def train_model(self, X_train, y_train):
        """训练XGBoost模型
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
        """
        # 如果需要参数调优
        if self.param_tuning:
            param_grid = {
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 2, 3],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'learning_rate': [0.01, 0.05, 0.1]
            }
            best_params = self.grid_search_cv(X_train, y_train, param_grid)
            self.params.update(best_params)
            logger.info("Using optimized parameters from grid search")
        
        # 计算正负类比例,用于scale_pos_weight
        neg_pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
        self.params['scale_pos_weight'] = neg_pos_ratio
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        
        logger.info(f"训练数据大小: {X_train.shape}")
        logger.info(f"训练集中的类别分布:")
        logger.info(f"  正样本数量: {np.sum(y_train == 1)}")
        logger.info(f"  负样本数量: {np.sum(y_train == 0)}")
        logger.info(f"  正样本权重: {self.params['scale_pos_weight']}")
        
        # 训练模型
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            verbose_eval=50
        )

    def grid_search_cv(self, X, y, param_grid):
        logger.info("开始网格搜索...")
        
        best_score = 0
        best_params = None
        
        # 转换为DMatrix格式
        dtrain = xgb.DMatrix(X, label=y)
        
        # 进行网格搜索
        for max_depth in param_grid['max_depth']:
            for min_child_weight in param_grid['min_child_weight']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        for learning_rate in param_grid['learning_rate']:
                            params = {
                                'max_depth': max_depth,
                                'min_child_weight': min_child_weight,
                                'subsample': subsample,
                                'colsample_bytree': colsample_bytree,
                                'learning_rate': learning_rate,
                                'objective': 'binary:logistic',
                                'eval_metric': 'auc',
                                'seed': 42
                            }
                            
                            # 使用交叉验证评估参数
                            cv_results = xgb.cv(
                                params,
                                dtrain,
                                num_boost_round=1000,
                                nfold=5,
                                early_stopping_rounds=50,
                                verbose_eval=False
                            )
                            
                            # 获取最佳得分
                            mean_score = cv_results['test-auc-mean'].max()
                            
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = params
                                
                            logger.info(f"参数: {params}")
                            logger.info(f"交叉验证分数: {mean_score:.4f}")
        
        logger.info(f"网格搜索完成。最佳参数: {best_params}")
        logger.info(f"最佳交叉验证分数: {best_score:.4f}")
        
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
        
        logger.info(f"\n=== {self.name}模型评估 ===")
        logger.info(f"测试数据大小: {X_test.shape}")
        
        # 获取预测概率
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = self.model.predict(dtest)
        
        # 找到最优阈值
        self.best_threshold = self.find_best_threshold(y_test, y_pred_proba)
        logger.info(f"使用阈值: {self.best_threshold:.4f}")
        
        # 使用最优阈值进行预测
        y_pred = (y_pred_proba > self.best_threshold).astype(int)
        
        test_metrics = self.metrics.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return test_metrics
                
    def predict(self, X_test):
        """预测类别"""
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = self.model.predict(dtest)
        return (y_pred_proba > self.best_threshold).astype(int)
        
    def predict_proba(self, X_test):
        """预测概率"""
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)

    def get_parameters(self):
        """获取模型参数"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # 保存XGBoost模型的关键参数
        return {
            'booster': self.model.save_raw(),  # 保存模型的原始数据
            'best_threshold': self.best_threshold,
            'params': self.params  # 保存训练参数
        }

    def set_parameters(self, parameters):
        """设置模型参数"""
        if 'booster' not in parameters:
            raise ValueError("Missing booster parameters")
        
        # 创建新的XGBoost模型
        self.model = xgb.Booster()
        # 从原始数据加载模型
        self.model.load_model(bytearray(parameters['booster']))
        
        # 恢复其他参数
        self.best_threshold = parameters.get('best_threshold', 0.5)
        self.params = parameters.get('params', self.params)

