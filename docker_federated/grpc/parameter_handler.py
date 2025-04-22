import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import pickle

try:
    from phe import paillier
    HAS_PHE = True
except ImportError:
    HAS_PHE = False

from src.utils.logging_config import get_logger

logger = get_logger()

class ParameterHandler(ABC):
    """参数处理的抽象基类"""
    
    @abstractmethod
    def serialize_parameters(self, parameters: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """序列化参数"""
        pass
        
    @abstractmethod
    def deserialize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """反序列化参数"""
        pass
        
    @abstractmethod
    def aggregate_parameters(self, parameters_list: list, weights: list) -> Dict[str, np.ndarray]:
        """聚合参数"""
        pass

class PlainParameterHandler(ParameterHandler):
    """普通参数处理类"""
    
    def serialize_parameters(self, parameters: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """直接序列化参数"""
        serialized = {}
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                serialized[key] = {
                    'data': value.tobytes(),
                    'shape': list(value.shape),
                    'dtype': str(value.dtype)
                }
            else:
                arr = np.array([value])
                serialized[key] = {
                    'data': arr.tobytes(),
                    'shape': [1],
                    'dtype': str(arr.dtype)
                }
        return serialized
        
    def deserialize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """直接反序列化参数"""
        deserialized = {}
        for key, value in parameters.items():
            try:
                dtype = np.dtype(value['dtype'])
                arr = np.frombuffer(value['data'], dtype=dtype).reshape(value['shape'])
                deserialized[key] = arr[0] if len(value['shape']) == 1 and value['shape'][0] == 1 else arr
            except Exception as e:
                logger.error(f"参数反序列化失败 {key}: {str(e)}")
                raise
        return deserialized
        
    def aggregate_parameters(self, parameters_list: list, weights: list) -> Dict[str, np.ndarray]:
        """直接聚合参数"""
        if not parameters_list or not weights:
            raise ValueError("参数列表或权重为空")
            
        aggregated = {}
        for key in parameters_list[0].keys():
            aggregated[key] = sum(p[key] * w for p, w in zip(parameters_list, weights))
        return aggregated

class HomomorphicParameterHandler(ParameterHandler):
    """同态加密参数处理类"""
    
    def __init__(self):
        if not HAS_PHE:
            raise ImportError("未安装phe库,无法使用同态加密功能")
            
        self.public_key = None
        self.private_key = None
        self._load_or_generate_keys()
        
    def _load_or_generate_keys(self):
        """加载或生成密钥对"""
        try:
            # 尝试从文件加载密钥
            key_dir = "/app/certs"
            public_key_path = f"{key_dir}/public_key.pkl"
            private_key_path = f"{key_dir}/private_key.pkl"
            
            if os.path.exists(public_key_path) and os.path.exists(private_key_path):
                with open(public_key_path, 'rb') as f:
                    self.public_key = pickle.load(f)
                with open(private_key_path, 'rb') as f:
                    self.private_key = pickle.load(f)
                logger.info("已加载现有密钥对")
            else:
                # 生成新密钥对
                self.public_key, self.private_key = paillier.generate_paillier_keypair()
                
                # 保存密钥
                os.makedirs(key_dir, exist_ok=True)
                with open(public_key_path, 'wb') as f:
                    pickle.dump(self.public_key, f)
                with open(private_key_path, 'wb') as f:
                    pickle.dump(self.private_key, f)
                logger.info("已生成并保存新密钥对")
                
        except Exception as e:
            logger.error(f"密钥处理失败: {str(e)}")
            raise
            
    def serialize_parameters(self, parameters: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """加密并序列化参数"""
        serialized = {}
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                # 将浮点数转换为整数进行加密
                scale = 1e6  # 保留6位小数
                value_int = (value * scale).astype(np.int64)
                encrypted = [self.public_key.encrypt(x) for x in value_int.flatten()]
                serialized[key] = {
                    'encrypted_data': pickle.dumps(encrypted),
                    'scale': scale,
                    'shape': list(value.shape),
                    'dtype': str(value.dtype)
                }
            else:
                arr = np.array([value])
                scale = 1e6
                value_int = (arr * scale).astype(np.int64)
                encrypted = [self.public_key.encrypt(x) for x in value_int]
                serialized[key] = {
                    'encrypted_data': pickle.dumps(encrypted),
                    'scale': scale,
                    'shape': [1],
                    'dtype': str(arr.dtype)
                }
        return serialized
        
    def deserialize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """解密并反序列化参数"""
        deserialized = {}
        for key, value in parameters.items():
            try:
                encrypted = pickle.loads(value['encrypted_data'])
                decrypted = [self.private_key.decrypt(x) for x in encrypted]
                dtype = np.dtype(value['dtype'])
                # 还原浮点数
                arr = (np.array(decrypted) / value['scale']).astype(dtype).reshape(value['shape'])
                deserialized[key] = arr[0] if len(value['shape']) == 1 and value['shape'][0] == 1 else arr
            except Exception as e:
                logger.error(f"参数解密失败 {key}: {str(e)}")
                raise
        return deserialized
        
    def aggregate_parameters(self, parameters_list: list, weights: list) -> Dict[str, np.ndarray]:
        """聚合加密参数"""
        if not parameters_list or not weights:
            raise ValueError("参数列表或权重为空")
            
        aggregated = {}
        for key in parameters_list[0].keys():
            # 对加密数据进行加权求和
            encrypted_sum = None
            scale = parameters_list[0][key]['scale']
            shape = parameters_list[0][key]['shape']
            dtype = parameters_list[0][key]['dtype']
            
            for params, weight in zip(parameters_list, weights):
                encrypted = pickle.loads(params[key]['encrypted_data'])
                weight_int = int(weight * 1e6)  # 将权重转换为整数
                if encrypted_sum is None:
                    encrypted_sum = [x * weight_int for x in encrypted]
                else:
                    encrypted_sum = [sum + (x * weight_int) for sum, x in zip(encrypted_sum, encrypted)]
            
            # 解密结果
            decrypted = [self.private_key.decrypt(x) for x in encrypted_sum]
            # 还原浮点数
            arr = (np.array(decrypted) / (scale * 1e6)).astype(np.dtype(dtype)).reshape(shape)
            aggregated[key] = arr
            
        return aggregated

def create_parameter_handler(use_encryption: bool = False) -> ParameterHandler:
    """创建参数处理器"""
    if use_encryption:
        if not HAS_PHE:
            logger.warning("未安装phe库,将使用普通模式")
            return PlainParameterHandler()
        try:
            return HomomorphicParameterHandler()
        except Exception as e:
            logger.error(f"创建加密处理器失败: {str(e)}, 将使用普通模式")
            return PlainParameterHandler()
    return PlainParameterHandler() 