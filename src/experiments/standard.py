from experiments.base_experiment import BaseExperiment
from utils.logging_config import get_logger
from tqdm import tqdm
import os
import pandas as pd

logger = get_logger()

class StandardExperiment(BaseExperiment):
    """标准训练实验类"""
    
    def __init__(self):
        super().__init__()
        
    def train_and_evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """训练并评估单个模型"""
        try:
            # 训练模型
            model.train_model(
                X_train=X_train, 
                y_train=y_train
            )
            
            # 评估模型
            metrics = model.evaluate_model(X_test, y_test)
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
                logger.info(f"训练模型: {name}")
                
                # 根据模型的normalize值选择合适的数据集
                if hasattr(model, 'normalize') and model.normalize:
                    if X_train_norm is None:
                        logger.warning(f"模型 {name} 需要标准化数据，但标准化数据不可用，跳过该模型")
                        continue
                    X_train, X_test, y_train, y_test = X_train_norm, X_test_norm, y_train_norm, y_test_norm
                    logger.info(f"使用标准化数据训练模型 {name}")
                else:
                    if X_train_raw is None:
                        logger.warning(f"模型 {name} 需要非标准化数据，但非标准化数据不可用，跳过该模型")
                        continue
                    X_train, X_test, y_train, y_test = X_train_raw, X_test_raw, y_train_raw, y_test_raw
                    logger.info(f"使用非标准化数据训练模型 {name}")
                
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
                    logger.info(f"\n最佳模型是 {best_model}, 得分: {best_score:.4f}")
                        
        except Exception as e:
            logger.error(f"模型对比失败: {str(e)}")
            raise 