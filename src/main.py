from utils.logging_config import setup_logging, get_logger
from experiments.standard import StandardExperiment
from experiments.federated import FederatedExperiment
from models.xgboost_model import XGBoostModel
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from models.neural_net import NeuralNetModel


logger = get_logger()

def run_standard_experiment(config_path):
    """运行标准训练实验"""
    experiment = StandardExperiment(config_path)
    
    # logger.info("添加XGBoost模型...")
    # experiment.add_model('xgboost', XGBoostModel(config=experiment.config))
    
    # logger.info("添加随机森林模型...")
    # experiment.add_model('random_forest', RandomForestModel(config=experiment.config))
    
    logger.info("添加逻辑回归模型...")
    experiment.add_model('logistic', LogisticRegressionModel(config=experiment.config))
    
    # logger.info("添加神经网络模型...")
    # experiment.add_model('neural_net', NeuralNetModel(config=experiment.config))
    
    # 运行实验
    experiment.run()
    
    # 比较模型性能
    experiment.compare_models()
    
def run_federated_experiment(config_path):
    """运行联邦学习实验"""
    experiment = FederatedExperiment(config_path)
    
    logger.info("添加逻辑回归模型...")
    experiment.add_model('logistic', LogisticRegressionModel(config=experiment.config))
    
    logger.info("添加神经网络模型...")
    experiment.add_model('neural_net', NeuralNetModel(config=experiment.config))
    
    # 运行联邦学习实验
    experiment.run(n_clients=3, n_rounds=10)

def main():
    config_path = 'config/default.yaml'
    setup_logging()
    
    logger.info("\n=== 开始实验 ===")
    
    # 运行标准训练实验
    logger.info("\n=== 运行标准训练实验 ===")
    run_standard_experiment(config_path)
    
    # 运行联邦学习实验
    # logger.info("\n=== 运行联邦学习实验 ===")
    # run_federated_experiment(config_path)

if __name__ == "__main__":
    main()