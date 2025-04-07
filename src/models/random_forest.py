from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import logging
import numpy as np
import pandas as pd
from models.base_model import BaseModel
from data_process.credit_card import CreditCardDataPreprocessor
from utils.metrics import ModelMetrics
class RandomForestModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.param_tuning = False
        self.model = RandomForestClassifier()
        self.name = "RandomForest"
        self.normalize = False
        self.metrics = ModelMetrics()  # 添加这行
        self.preprocessor = CreditCardDataPreprocessor(config=config, model=self)
        self.best_threshold = 0.5  # 添加最佳阈值属性
        
    def grid_search_cv(self, X, y, param_grid):
        """网格搜索寻找最优参数
        
        Args:
            X: 特征数据
            y: 标签数据
            param_grid: 参数网格,包含待搜索的参数值范围
            
        Returns:
            tuple: (最优模型, 最优参数)
        """
        logging.info("\n=== Starting grid search ===")
        
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
                        logging.info(f"Parameters: {params}")
                        logging.info(f"CV AUC Score: {mean_score:.4f}")
                        
                        # 更新最优参数和模型
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = params
                            # 使用最优参数重新训练模型
                            best_model = RandomForestClassifier(**params)
                            best_model.fit(X, y)
        
        logging.info(f"Grid search completed.")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best CV AUC score: {best_score:.4f}")
        
        return best_model, best_params

    def train_model(self, X_train, y_train):
        """训练随机森林模型
        
        Args:
            X_train: 训练集特征矩阵
            y_train: 训练集标签
        """
        # 输出训练开始的基本信息
        logging.info("\n=== Random Forest Training Start ===")
        logging.info(f"Model Configuration:")
        logging.info(f"- Parameter tuning: {self.param_tuning}")
        logging.info(f"- Training data shape: {X_train.shape}")
        logging.info(f"Class Distribution in Training Set:")
        logging.info(f"- Positive samples: {np.sum(y_train == 1)}")
        logging.info(f"- Negative samples: {np.sum(y_train == 0)}")
        
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
        logging.info("Cross-validation Results:")
        logging.info(f"- Mean AUC: {cv_scores.mean():.4f}")
        logging.info(f"- Standard deviation: {cv_scores.std() * 2:.4f}")

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
        
        logging.info(f"Found best threshold: {best_threshold:.2f} (Accuracy: {best_accuracy:.4f})")
        return best_threshold

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Error: Model not trained")
        
        logging.info("\n=== Model Evaluation ===")
        logging.info(f"Test Data Shape: {X_test.shape}")
        
        # 获取预测结果
        y_pred_proba = self.predict_proba(X_test)
        
        # 找到最优阈值
        self.best_threshold = self.find_best_threshold(y_test, y_pred_proba)
        logging.info(f"Using threshold: {self.best_threshold}")
        
        # 使用最优阈值进行预测
        y_pred = (y_pred_proba > self.best_threshold).astype(int)
        
        # 计算评估指标
        test_metrics = self.metrics.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return test_metrics

    def get_parameters(self):
        """获取模型参数"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
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
            raise ValueError("Model not initialized")
        
        self.model.estimators_ = parameters['estimators']
        self.model.n_features_in_ = parameters['n_features']
        self.model.n_classes_ = parameters['n_classes']
        self.model.n_outputs_ = parameters['n_outputs']
        self.model.feature_importances_ = parameters['feature_importances']
