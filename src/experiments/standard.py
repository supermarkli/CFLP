from experiments.base import BaseExperiment
from utils.logging_config import get_logger
from tqdm import tqdm

logger = get_logger()

class StandardExperiment(BaseExperiment):
    """标准训练实验类"""
    
    def __init__(self, config_path):
        super().__init__(config_path)
        
    def train_and_evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """训练并评估单个模型"""
        try:
            # 训练模型
            model.train_model(
                X_train=X_train, 
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
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
            df = self.load_data()
            
            for name, model in tqdm(self.models.items(), desc="训练模型"):
                logger.info(f"处理模型数据: {name}")
                
                X_train, X_val, X_test, y_train, y_val, y_test = model.preprocess_data(df)
                
                metrics = self.train_and_evaluate_model(
                    model,
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test
                )
                self.metrics.add_model_metrics(name, metrics)
                
                # 保存模型
                if self.config.get('save_models', False):
                    model_path = f"{self.config['model_save_path']}/{name}.model"
                    self.save_model(model, model_path)
                
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