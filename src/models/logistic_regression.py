from sklearn.linear_model import SGDClassifier
import logging
from models.base_model import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from utils.metrics import ModelMetrics
from data_process.credit_card import CreditCardDataPreprocessor

class LogisticRegressionModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.param_tuning = False
        self.model = None
        self.metrics = ModelMetrics()
        self.name = "LogisticRegression"
        self.normalize = True
        self.best_threshold = 0.5
        self.preprocessor = CreditCardDataPreprocessor(config=config, model=self)
        
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
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
        else:
            self.model = SGDClassifier(
                loss='log_loss',
                eta0=0.1,
                learning_rate='adaptive',
                max_iter=1000,
                class_weight='balanced'
            )
        
        logging.info("\n=== Training Start ===")
        logging.info(f"Training data size: {X_train.shape}")
        logging.info(f"Class distribution in training set:")
        logging.info(f"  Positive samples: {np.sum(y_train == 1)}")
        logging.info(f"  Negative samples: {np.sum(y_train == 0)}")
        
        if X_val is not None:
            logging.info(f"Validation data size: {X_val.shape}")
            
        # 训练模型
        self.model.fit(X_train, y_train)
        logging.info("Model training completed")
        
        # 在验证集上找最优阈值
        if X_val is not None and y_val is not None:
            self.find_best_threshold(X_val, y_val)
            
    def grid_search_cv(self, X, y, param_grid):
        """网格搜索寻找最优参数"""
        logging.info("Starting grid search...")
        
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
                        logging.info(f"Parameters: {params}")
                        logging.info(f"Cross-validation score: {mean_score:.4f}")
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = params
        
        logging.info(f"Grid search completed. Best parameters: {best_params}")
        logging.info(f"Best cross-validation score: {best_score:.4f}")
        
        return best_params
        
    def find_best_threshold(self, X_val, y_val):
        """找到最优的预测阈值"""
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # 尝试不同的阈值
        thresholds = np.arange(0.2, 0.7, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.best_threshold = best_threshold
        logging.info(f"Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
        
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("Model not trained")
            
        logging.info("\n=== Model Evaluation ===")
        logging.info(f"Using threshold: {self.best_threshold}")
        

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > self.best_threshold).astype(int)
        
        # 计算评估指标
        test_metrics = self.metrics.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # 分析并输出特征重要性
        # self.analyze_feature_importance()
        
        return test_metrics
        
    def analyze_feature_importance(self):
        """分析特征重要性"""
        if self.model is None:
            logging.error("Model not trained")
            return
        
        try:
            # 获取特征重要性
            importances = np.abs(self.model.coef_[0])
            
            # 如果没有feature_names，则使用数字索引
            if not hasattr(self, 'feature_names') or self.feature_names is None:
                self.feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # 确保长度匹配
            if len(self.feature_names) != len(importances):
                logging.warning(f"Feature names length ({len(self.feature_names)}) "
                              f"doesn't match coefficients length ({len(importances)}). "
                              "Using numeric feature names.")
                self.feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # 输出前10个重要特征
            logging.info("Top 10 important features:")
            for idx, row in importance_df.head(10).iterrows():
                logging.info(f"{row['feature']}: {row['importance']:.4f}")
            
        except Exception as e:
            logging.error(f"Error in analyzing feature importance: {str(e)}")

    def predict(self, X_test):

        return self.model.predict(X_test)

    def predict_proba(self, X_test):

        return self.model.predict_proba(X_test)[:, 1]  # 取正类概率

    def get_parameters(self):
        """获取模型参数"""
        return {
            'coef': self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy()
        }
        
    def set_parameters(self, parameters):
        """设置模型参数"""
        self.model.coef_ = parameters['coef']
        self.model.intercept_ = parameters['intercept']
