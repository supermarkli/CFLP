import logging
from datetime import datetime

def setup_logging():
    """设置简单的控制台日志配置"""
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 如果logger已经有处理器,不重复添加
    if logger.handlers:
        return logger
        
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(console_handler)
    
    return logger

def get_logger():
    """获取logger实例"""
    logger = logging.getLogger()
    if not logger.handlers:
        logger = setup_logging()
    return logger
