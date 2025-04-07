import xgboost as xgb
import pandas as pd  
from sklearn.model_selection import cross_val_score
import logging
from models.base_model import BaseModel
from utils.metrics import ModelMetrics  
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

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
            logging.info("Using optimized parameters from grid search")
        
        # 计算正负类比例,用于scale_pos_weight
        neg_pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
        self.params['scale_pos_weight'] = neg_pos_ratio
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        
        logging.info("\n=== Training Start ===")
        logging.info(f"Training data size: {X_train.shape}")
        logging.info(f"Class distribution in training set:")
        logging.info(f"  Positive samples: {np.sum(y_train == 1)}")
        logging.info(f"  Negative samples: {np.sum(y_train == 0)}")
        logging.info(f"  Scale pos weight: {self.params['scale_pos_weight']}")
        
        # 训练模型
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            verbose_eval=50
        )

    def grid_search_cv(self, X, y, param_grid):
        logging.info("Starting grid search...")
        
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
                                
                            logging.info(f"Parameters: {params}")
                            logging.info(f"Cross-validation score: {mean_score:.4f}")
        
        logging.info(f"Grid search completed. Best parameters: {best_params}")
        logging.info(f"Best cross-validation score: {best_score:.4f}")
        
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
        
        logging.info(f"Found best threshold: {best_threshold:.2f} (Accuracy: {best_accuracy:.4f})")
        return best_threshold
        
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        logging.info("\n=== Model Evaluation ===")
        logging.info(f"Test data size: {X_test.shape}")
        
        # 获取预测概率
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = self.model.predict(dtest)
        
        # 找到最优阈值
        self.best_threshold = self.find_best_threshold(y_test, y_pred_proba)
        logging.info(f"Using threshold: {self.best_threshold}")
        
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

