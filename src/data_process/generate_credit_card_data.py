import os
import sys

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 添加项目根目录到Python路径
sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.utils.logging_config import setup_logging, get_logger
import yaml

logger = get_logger()

def load_config():
    """加载配置文件"""
    # 修改配置文件路径
    config_path = os.path.join(PROJECT_ROOT, 'src', 'config', 'default.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def clean_data(df):
    """数据清洗"""
    X = df.copy()
    original_shape = X.shape
    
    # 性别映射 (1->0, 2->1)
    gender_mapping = {1: 0, 2: 1}
    X['SEX'] = X['SEX'].map(gender_mapping)
    
    # 教育程度重新映射
    X['EDUCATION'] = X['EDUCATION'].replace([1,3,4,5,6], [3,1,0,0,0])
    
    # 婚姻状态映射
    X['MARRIAGE'] = X['MARRIAGE'].replace(3, 0)
    
    # 还款状况映射
    for i in range(1, 7):
        X[f'PAY_{i}'] = X[f'PAY_{i}'].replace([-1, -2], [0, 0])
        
    # 处理缺失值
    X = X.dropna()
    logger.info(f"数据清洗完成: 原始形状 {original_shape} -> 清洗后形状 {X.shape}")
    return X

def split_features_target(df):
    """分离特征和目标变量"""
    X = df.drop(['ID', 'default payment next month'], axis=1)
    y = df['default payment next month']
    logger.info(f"特征分离完成: 特征数量 {X.shape[1]}, 样本数量 {X.shape[0]}")
    logger.info(f"目标变量分布: \n{y.value_counts(normalize=True)}")
    return X, y

def preprocess_data(normalize=True):
    """数据预处理(标准化和独热编码)"""
    # 定义数值型和分类型特征
    numeric_features = ['LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                       'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    logger.info(f"数值型特征: {len(numeric_features)}个")
    logger.info(f"分类型特征: {len(categorical_features)}个")
    
    if normalize:
        transformers = [
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
        logger.info("使用标准化和独热编码进行特征转换")
    else:
        all_features = numeric_features + categorical_features
        transformers = [('passthrough', 'passthrough', all_features)]
        logger.info("使用原始特征值")
    
    # 创建并应用预处理器
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

def save_data(X, y, output_dir, prefix):
    """保存数据到csv文件"""
    if isinstance(X, np.ndarray):
        columns = [f'feature_{i}' for i in range(X.shape[1])]
        data = pd.DataFrame(X, columns=columns)
    else:
        data = X.copy()

    if isinstance(y, pd.Series) and len(y) == len(data):
        data['target'] = y.values  
    else:
        data['target'] = y  
    
    output_path = os.path.join(output_dir, f'credit_card_{prefix}.csv')
    data.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # 转换为MB
    logger.info(f"数据已保存到: {output_path}")
    logger.info(f"文件大小: {file_size:.2f}MB")
    return len(data)

def generate_credit_card_data():
    # 加载配置
    config = load_config()
    logger.info("成功加载配置文件")
    
    # 读取原始数据
    data_path = config['data_path']
    abs_data_path = os.path.join(PROJECT_ROOT, data_path)
    df = pd.read_csv(abs_data_path, skiprows=1)
    logger.info(f"成功加载数据,形状: {df.shape}")
    
    # 数据清洗
    df_cleaned = clean_data(df)
    
    # 分离特征和目标变量
    X, y = split_features_target(df_cleaned)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'],
        random_state=config['base']['random_seed']
    )
    
    logger.info(f"数据集划分完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    
    # 创建输出目录
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'credit_card')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"创建输出目录: {output_dir}")
    
    # 处理并保存标准化数据
    preprocessor_norm = preprocess_data(normalize=True)
    X_train_norm = preprocessor_norm.fit_transform(X_train)
    X_test_norm = preprocessor_norm.transform(X_test)
    
    logger.info("完成标准化数据处理")
    
    train_size_norm = save_data(X_train_norm, y_train, output_dir, 'train_normalized')
    test_size_norm = save_data(X_test_norm, y_test, output_dir, 'test_normalized')
    
    # 处理并保存非标准化数据
    preprocessor_raw = preprocess_data(normalize=False)
    X_train_raw = preprocessor_raw.fit_transform(X_train)
    X_test_raw = preprocessor_raw.transform(X_test)
    logger.info("完成非标准化数据处理")
    
    train_size_raw = save_data(X_train_raw, y_train, output_dir, 'train_raw')
    test_size_raw = save_data(X_test_raw, y_test, output_dir, 'test_raw')
    
    logger.info("数据生成完成!")
    logger.info("标准化数据:")
    logger.info(f"训练集大小: {train_size_norm}")
    logger.info(f"测试集大小: {test_size_norm}")
    logger.info("\n非标准化数据:")
    logger.info(f"训练集大小: {train_size_raw}")
    logger.info(f"测试集大小: {test_size_raw}")

if __name__ == '__main__':
    generate_credit_card_data() 