import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import pandas as pd
from tqdm import tqdm

from src.experiments.base_experiment import BaseExperiment
from src.utils.logging_config import get_logger

logger = get_logger()

class StandardExperiment(BaseExperiment):
    """标准训练实验类"""
    
    def __init__(self):
        super().__init__()
        
    def train_and_evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """训练并评估单个模型"""
        try:
            
            model.train_model(
                X_train=X_train, 
                y_train=y_train
            )
            
            metrics = model.evaluate_model(X_test, y_test)
            logger.info(f"{model.name}模型评估指标:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1']:.4f}")
            logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"模型 {model.name} 训练失败: {str(e)}")
            raise

    def run(self):
        """运行实验"""
        try:
            # 加载数据
            X_train_norm, X_test_norm, y_train_norm, y_test_norm, X_train_raw, X_test_raw, y_train_raw, y_test_raw = self.load_data()
            
            for name, model in tqdm(self.models.items(), desc="训练模型"):
                logger.info(f"\n=== {name} 模型训练  ===")
                
                # 根据模型的normalize值选择合适的数据集
                if hasattr(model, 'normalize') and model.normalize:
                    if X_train_norm is None:
                        logger.warning(f" {name} 模型需要标准化数据，但标准化数据不可用，跳过该模型")
                        continue
                    X_train, X_test, y_train, y_test = X_train_norm, X_test_norm, y_train_norm, y_test_norm
                    logger.info(f"使用标准化数据训练 {name} 模型 ")
                else:
                    if X_train_raw is None:
                        logger.warning(f" {name} 模型需要非标准化数据，但非标准化数据不可用，跳过该模型")
                        continue
                    X_train, X_test, y_train, y_test = X_train_raw, X_test_raw, y_train_raw, y_test_raw
                    logger.info(f"使用非标准化数据训练 {name} 模型 ")
                
                # 将DataFrame转换为NumPy数组
                if isinstance(X_train, pd.DataFrame):
                    X_train = X_train.to_numpy()
                if isinstance(X_test, pd.DataFrame):
                    X_test = X_test.to_numpy()
                if isinstance(y_train, pd.Series):
                    y_train = y_train.to_numpy()
                if isinstance(y_test, pd.Series):
                    y_test = y_test.to_numpy()
                
                # 训练并评估模型
                metrics = self.train_and_evaluate_model(
                    model,
                    X_train, y_train,
                    X_test, y_test
                )
                self.metrics.add_model_metrics(name, metrics)
                
        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")
            raise
            
    def compare_models(self):
        """比较所有模型的性能"""
        try:
            results = self.metrics.compare_models()
            if results is not None:
                # 显示模型对比表格
                logger.info("\n=== 模型性能对比 ===")
                logger.info("\n" + results.to_string())
                
                # 获取最佳模型
                best_model, best_score = self.metrics.get_best_model()
                if best_model:
                    logger.info(f"最佳模型是 {best_model}, 得分: {best_score:.4f}")
                        
        except Exception as e:
            logger.error(f"模型对比失败: {str(e)}")
            raise 