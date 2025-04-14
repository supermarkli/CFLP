import os
import sys
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.utils.logging_config import setup_logging, get_logger


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

def split_data_for_federation(num_clients=3):
    """将标准化训练数据分割成多个部分，为联邦学习做准备"""
    logger.info(f"开始为 {num_clients} 个客户端分割标准化训练数据...")

    # 定义路径
    base_data_dir = os.path.join(PROJECT_ROOT, 'data')
    source_file_path = os.path.join(base_data_dir, 'credit_card', 'credit_card_train_normalized.csv')

    # 加载数据
    try:
        df_train_norm = pd.read_csv(source_file_path)
        logger.info(f"成功加载标准化训练数据: {source_file_path}, 形状: {df_train_norm.shape}")
    except FileNotFoundError:
        logger.error(f"源文件未找到: {source_file_path}。请先运行 generate_credit_card_data()")
        return
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        return

    # 检查数据是否为空
    if df_train_norm.empty:
        logger.error("加载的数据为空，无法进行分割。")
        return

    # 计算分割点
    n_samples = len(df_train_norm)
    indices = np.arange(n_samples)
    # 可以选择随机打乱数据再分割
    np.random.shuffle(indices) 
    split_indices = np.array_split(indices, num_clients)
    logger.info(f"数据将被分割成 {len(split_indices)} 份。")


    # 分割并保存数据
    for i in range(num_clients):
        client_id = i + 1
        client_dir = os.path.join(base_data_dir, f'client{client_id}')
        os.makedirs(client_dir, exist_ok=True) # 创建客户端数据目录

        # 获取当前客户端的数据子集
        client_data_indices = split_indices[i]
        if len(client_data_indices) == 0:
             logger.warning(f"客户端 {client_id} 分配到的数据为空，跳过保存。")
             continue

        client_df = df_train_norm.iloc[client_data_indices]

        # 定义客户端数据保存路径 (使用相同的文件名)
        client_output_path = os.path.join(client_dir, 'credit_card_train_normalized.csv')

        # 保存数据
        try:
            client_df.to_csv(client_output_path, index=False)
            file_size = os.path.getsize(client_output_path) / (1024 * 1024) # MB
            logger.info(f"成功保存客户端 {client_id} 的数据到: {client_output_path}, 形状: {client_df.shape}, 大小: {file_size:.2f}MB")
        except Exception as e:
            logger.error(f"保存客户端 {client_id} 数据时出错: {e}")


    logger.info(f"为 {num_clients} 个客户端分割数据完成。")


if __name__ == '__main__':
    # 配置日志记录 (新增)
    setup_logging()
    logger.info("开始执行数据处理脚本...") # (新增)

    # 1. 生成信用卡数据（调用你原有的函数）
    generate_credit_card_data()

    # 2. 将标准化训练数据分割给客户端 (新增调用)
    split_data_for_federation(num_clients=3)

    logger.info("数据处理脚本执行完毕。") # (新增) 