from utils.logging_config import setup_logging, get_logger
from experiments.standard import StandardExperiment
from experiments.federated import FederatedExperiment
from models.xgboost_model import XGBoostModel
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from models.neural_net import NeuralNetModel


logger = get_logger()

def run_standard_experiment():
    """运行标准训练实验"""
    experiment = StandardExperiment()
    
    logger.info("添加XGBoost模型...")
    experiment.add_model('xgboost', XGBoostModel())
    
    logger.info("添加随机森林模型...")
    experiment.add_model('random_forest', RandomForestModel())
    
    logger.info("添加逻辑回归模型...")
    experiment.add_model('logistic', LogisticRegressionModel())
    
    logger.info("添加神经网络模型...")
    experiment.add_model('neural_net', NeuralNetModel())
    
    # 运行实验
    experiment.run()
    
    # 比较模型性能
    experiment.compare_models()
    
def run_federated_experiment():
    """运行联邦学习实验"""
    experiment = FederatedExperiment()
    
    logger.info("添加逻辑回归模型...")
    experiment.add_model('logistic', LogisticRegressionModel())
    
    logger.info("添加神经网络模型...")
    experiment.add_model('neural_net', NeuralNetModel())
    
    # 运行联邦学习实验
    experiment.run(n_clients=3, n_rounds=10)

def main():
    setup_logging()
        
    # 运行标准训练实验
    logger.info("\n=== 运行标准训练实验 ===")
    run_standard_experiment()
    
    # 运行联邦学习实验
    logger.info("\n=== 运行联邦学习实验 ===")
    run_federated_experiment()

if __name__ == "__main__":
    main()