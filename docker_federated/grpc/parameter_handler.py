import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Any, Tuple, Optional, Union, List
from abc import ABC, abstractmethod
import pickle
from phe import paillier
from docker_federated.grpc.generated import federation_pb2
from src.utils.logging_config import get_logger

logger = get_logger()

class ParameterHandler:
    """统一的参数处理器"""
    
    def __init__(self, is_client: bool = False, use_encryption: bool = False):
        self.is_client = is_client
        self.use_encryption = use_encryption
        self.SCALE_FACTOR = 1e6
        
        if use_encryption:
            self._load_keys()
            
    def _load_keys(self):
        """加载密钥"""
        try:
            if not self.is_client:
                # 服务器加载私钥
                with open('/app/certs/private_key.pkl', 'rb') as f:
                    self.private_key = pickle.load(f)
                logger.info("服务器已加载私钥")
            else:
                # 客户端加载公钥
                with open('/app/certs/public_key.pkl', 'rb') as f:
                    self.public_key = pickle.load(f)
                logger.info("客户端已加载公钥")
        except Exception as e:
            logger.error(f"密钥加载失败: {str(e)}")
            raise
            
    def serialize_plain_parameters(self, parameters: dict) -> dict:
        serialized = {}
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                numpy_array = federation_pb2.NumpyArray(
                    data=value.tobytes(),
                    shape=list(value.shape),
                    dtype=str(value.dtype)
                )
                serialized[key] = numpy_array
            else:
                arr = np.array([value])
                numpy_array = federation_pb2.NumpyArray(
                    data=arr.tobytes(),
                    shape=[1],
                    dtype=str(arr.dtype)
                )
                serialized[key] = numpy_array
        return serialized

    def serialize_encrypted_parameters(self, parameters: dict) -> dict:
        """
        加密序列化：输入dict[str, np.ndarray/int/float/标量]，输出dict[str, NumpyArray]（dtype='encrypted'）
        """
        result = {}
        for key, value in parameters.items():
            logger.debug(f"[serialize_encrypted_parameters] key={key}, original type={type(value)}, value={value}")
            value = np.array(value, dtype=np.float64)
            if value.ndim == 0:
                value = value.reshape(1)
            logger.debug(f"[serialize_encrypted_parameters] key={key}, after np.array: type={type(value)}, dtype={value.dtype}, shape={value.shape}, value={value}")
            value_int = (value * self.SCALE_FACTOR).astype(np.int64)
            encrypted = [self.public_key.encrypt(x) for x in value_int.flatten()]
            numpy_array = federation_pb2.NumpyArray(
                data=pickle.dumps(encrypted),
                shape=list(value.shape),
                dtype="encrypted"
            )
            result[key] = numpy_array
        return result

    def deserialize_plain_parameters(self, parameters: dict) -> dict:
        """
        明文反序列化：输入dict[str, NumpyArray]，输出dict[str, np.ndarray]
        """
        result = {}
        param_mapping = {
            'weights': 'coef_',
            'bias': 'intercept_'
        }
        for key, value in parameters.items():
            mapped_key = param_mapping.get(key, key)
            dtype = np.dtype(value.dtype)
            arr = np.frombuffer(value.data, dtype=dtype).reshape(list(value.shape))
            result[mapped_key] = arr
        return result

    def deserialize_encrypted_parameters(self, parameters: dict) -> dict:
        """
        加密反序列化：输入dict[str, NumpyArray]，输出dict[str, np.ndarray]
        """
        result = {}
        for key, value in parameters.items():
            encrypted = pickle.loads(value.data)
            decrypted = [self.private_key.decrypt(x) for x in encrypted]
            arr = (np.array(decrypted) / self.SCALE_FACTOR).astype(np.float64)
            arr = arr.reshape(list(value.shape))
            result[key] = arr
        return result

    def encrypt(self, value: Union[float, np.ndarray]) -> bytes:
        """加密单个值或数组"""
        if not self.is_client:
            raise ValueError("只有客户端可以加密数据")
            
        if not isinstance(value, np.ndarray):
            value = np.array([value])
            
        value_int = (value * self.SCALE_FACTOR).astype(np.int64)
        encrypted = [self.public_key.encrypt(x) for x in value_int.flatten()]
        return pickle.dumps(encrypted)
        
    def decrypt(self, encrypted_data: bytes, shape: tuple, dtype: str, scale: float) -> np.ndarray:
        """解密数据"""
        if self.is_client:
            raise ValueError("只有服务器可以解密数据")
            
        try:
            encrypted = pickle.loads(encrypted_data)
            decrypted = [self.private_key.decrypt(x) for x in encrypted]
            arr = (np.array(decrypted) / scale).astype(np.dtype(dtype))
            return arr.reshape(shape)
        except Exception as e:
            logger.error(f"解密失败: {str(e)}")
            raise

    def aggregate_parameters(self, parameters_list: List[Dict[str, Any]], weights: List[float] = None) -> Dict[str, np.ndarray]:
        """聚合参数列表
        
        Args:
            parameters_list: 参数列表，每个元素是一个参数字典
            weights: 权重列表，用于加权平均
            
        Returns:
            聚合后的参数字典
        """
        try:
            if not parameters_list:
                raise ValueError("参数列表为空")
            
            if weights is None:
                weights = [1.0 / len(parameters_list)] * len(parameters_list)
            
            if len(weights) != len(parameters_list):
                raise ValueError("权重列表长度与参数列表长度不匹配")
            
            if not self.use_encryption:
                return self._aggregate_plain(parameters_list, weights)
            else:
                return self._aggregate_encrypted(parameters_list, weights)
            
        except Exception as e:
            logger.error(f"参数聚合失败: {str(e)}")
            raise

    def _aggregate_plain(self, parameters_list: List[Dict[str, Any]], weights: List[float]) -> Dict[str, np.ndarray]:
        """聚合普通参数"""
        try:
            aggregated = {}
            # 获取所有参数的键
            keys = set()
            for params in parameters_list:
                keys.update(params.keys())
            
            # 对每个参数进行加权平均
            for key in keys:
                # 收集所有参数值
                values = []
                for params in parameters_list:
                    if key in params:
                        values.append(params[key])
                
                if not values:
                    continue
                
                # 确保所有值都是numpy数组
                values = [np.array(v) if not isinstance(v, np.ndarray) else v for v in values]
                
                # 检查形状是否一致
                shapes = [v.shape for v in values]
                if not all(s == shapes[0] for s in shapes):
                    raise ValueError(f"参数 {key} 的形状不一致")
                
                # 加权平均
                weighted_sum = np.zeros_like(values[0])
                for v, w in zip(values, weights):
                    weighted_sum += v * w
                
                aggregated[key] = weighted_sum
            
            return aggregated
        
        except Exception as e:
            logger.error(f"普通参数聚合失败: {str(e)}")
            raise

    def _aggregate_encrypted(self, parameters_list: List[Dict[str, Any]], weights: List[float]) -> Dict[str, np.ndarray]:
        """聚合加密参数"""
        try:
            if self.is_client:
                raise ValueError("客户端不能聚合加密参数")
            
            aggregated = {}
            # 获取所有参数的键
            keys = set()
            for params in parameters_list:
                keys.update(params.keys())
            
            # 对每个参数进行加权求和
            for key in keys:
                # 收集所有加密参数
                encrypted_values = []
                scales = []
                shapes = []
            
                for params in parameters_list:
                    if key in params:
                        value = params[key]
                        if 'encrypted_data' not in value:
                            raise ValueError(f"参数 {key} 不是加密参数")
                        
                        encrypted_values.append(pickle.loads(value['encrypted_data']))
                        scales.append(value['scale'])
                        shapes.append(value['shape'])
                
                if not encrypted_values:
                    continue
                
                # 检查形状是否一致
                if not all(s == shapes[0] for s in shapes):
                    raise ValueError(f"参数 {key} 的形状不一致")
                
                # 检查缩放因子是否一致
                if not all(s == scales[0] for s in scales):
                    raise ValueError(f"参数 {key} 的缩放因子不一致")
                
                # 加权求和
                weighted_sum = []
                for i, encrypted in enumerate(encrypted_values):
                    for j, val in enumerate(encrypted):
                        if i == 0:
                            weighted_sum.append(val * weights[i])
                        else:
                            weighted_sum[j] += val * weights[i]
                
                # 解密并还原
                decrypted = [self.private_key.decrypt(x) for x in weighted_sum]
                arr = np.array(decrypted) / scales[0]
                arr = arr.reshape(shapes[0])
                
                aggregated[key] = arr
            
            return aggregated
        
        except Exception as e:
            logger.error(f"加密参数聚合失败: {str(e)}")
            raise

def create_parameter_handler(use_encryption: bool = False, is_client: bool = True) -> ParameterHandler:
    """创建参数处理器
    
    Args:
        use_encryption: 是否使用加密
        is_client: 是否为客户端，客户端只需要公钥，服务端需要私钥
        
    Returns:
        ParameterHandler: 参数处理器实例
    """
    return ParameterHandler(is_client, use_encryption)
