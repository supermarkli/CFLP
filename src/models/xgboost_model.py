import xgboost as xgb
import pandas as pd  
from sklearn.model_selection import cross_val_score
import logging
from models.base_model import BaseModel
from utils.metrics import ModelMetrics  
import numpy as np
from sklearn.metrics import f1_score
from data_process.credit_card import CreditCardDataPreprocessor

class XGBoostModel(BaseModel):
    """XGBoost模型类
    
    实现了基于XGBoost的二分类模型,包含:
    1. 模型训练与预测
    2. 参数优化
    3. 模型评估
    4. 特征重要性分析
    """
    def __init__(self, config):
        """初始化XGBoost模型
        
        Args:
            config: 配置字典,包含模型参数等配置信息
            
        初始化内容:
        1. 基础参数设置
        2. 模型实例创建
        3. 训练参数配置
        """
        super().__init__(config)
        self.param_tuning = False
        self.model = None
        self.metrics = ModelMetrics()  # 初始化评估器
        self.name = "XGBoost"  # 添加模型名称属性
        self.best_threshold = 0.5  # 添加最优阈值属性
        self.normalize = False
        self.preprocessor = CreditCardDataPreprocessor(config=config, model=self)
        
        # 调整参数以提高召回率
        self.params = {
            'max_depth': 4,             # 略微增加深度
            'eta': 0.01,
            'objective': 'binary:logistic',
            'eval_metric': ['auc'],  
            'min_child_weight': 2,      # 降低以允许更多分裂
            # 'gamma': 0.1,               # 降低分裂阈值
            'subsample': 0.8,           # 增加采样比例
            'colsample_bytree': 0.7,   # 增加特征采样
        }
        
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """训练XGBoost模型
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征,可选
            y_val: 验证集标签,可选
            param_tuning: 是否进行参数调优,默认False
        """
        # 如果需要参数调优
        if self.param_tuning :
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
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'validation')]
        else:
            evals = [(dtrain, 'train')]
        
        logging.info("\n=== Training Start ===")
        logging.info(f"Training data size: {X_train.shape}")
        logging.info(f"Class distribution in training set:")
        logging.info(f"  Positive samples: {np.sum(y_train == 1)}")
        logging.info(f"  Negative samples: {np.sum(y_train == 0)}")
        logging.info(f"  Scale pos weight: {self.params['scale_pos_weight']}")
        
        if X_val is not None:
            logging.info(f"Validation data size: {X_val.shape}")
        
        # 训练模型
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=50
        )
        
        # 在验证集上找最优阈值
        if X_val is not None and y_val is not None:
            self.find_best_threshold(X_val, y_val)

    def find_best_threshold(self, X_val, y_val):
        """找到最优的预测阈值"""
        dval = xgb.DMatrix(X_val)
        y_pred_proba = self.model.predict(dval)
        
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

    def grid_search_cv(self, X, y, param_grid):
        """网格搜索寻找最优参数
        
        Args:
            X: 特征数据
            y: 标签数据
            param_grid: 参数网格,包含待搜索的参数值范围
            
        Returns:
            dict: 最优参数组合
            
        实现步骤:
        1. 将数据转换为DMatrix格式
        2. 对参数组合进行网格搜索
        3. 使用5折交叉验证评估每组参数
        4. 记录并返回最优参数组合
        """
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
        
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        logging.info("\n=== Model Evaluation ===")
        logging.info(f"Test data size: {X_test.shape}")
        logging.info(f"Using threshold: {self.best_threshold}")
        
        # 使用最优阈值进行预测
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba > self.best_threshold).astype(int)
        
        test_metrics = self.metrics.calculate_metrics(y_test, y_pred, y_pred_proba)

        # 分析并输出特征重要性
        # self.analyze_feature_importance()
        
        return test_metrics
        
    def analyze_feature_importance(self):
        """分析特征重要性"""
        if self.model is None or self.feature_names is None:
            logging.error("Model not trained or feature names not set")
            return
            
        # 直接使用get_score()获取特征重要性
        importance_dict = self.model.get_score(importance_type='weight')
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # 输出前10个重要特征
        logging.info("Top 10 important features:")
        for idx, row in importance_df.head(10).iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.4f}")
            
    @staticmethod
    def load_model(model_path):
        """加载已保存的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            XGBoost模型实例
            
        异常处理:
        捕获并记录加载过程中的任何错误
        """
        try:
            model = xgb.Booster()
            model.load_model(model_path)
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self, X_test):
        """预测类别
        
        Args:
            X_test: 测试数据特征矩阵
            
        Returns:
            预测的类别标签(0或1)
        """
        dtest = xgb.DMatrix(X_test)
        return (self.model.predict(dtest) > 0.5).astype(int)
        
    def predict_proba(self, X_test):
        """预测概率
        
        Args:
            X_test: 测试数据特征矩阵
            
        Returns:
            预测的正类概率
        """
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)

