import numpy as np
import logging
from phe import paillier  # 使用Paillier同态加密库

# 设置日志记录
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为DEBUG，记录所有级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式：时间戳 - 日志级别 - 消息内容
    handlers=[
        logging.StreamHandler(),  # 将日志输出到控制台
        logging.FileHandler("federated_learning.log")  # 将日志保存到文件
    ]
)

logger = logging.getLogger(__name__)  # 获取日志记录器实例

class Aggregator:
    """
    联邦学习中的聚合器类，用于处理安全聚合。
    支持Paillier同态加密方式。
    """

    def __init__(self, method="paillier"):
        """
        初始化聚合器
        :param method: 安全聚合方法，目前支持"paillier"
        """
        self.method = method
        
        if self.method == "paillier":
            # 生成Paillier公钥和私钥对
            self.public_key, self.private_key = paillier.generate_paillier_keypair()
        else:
            raise ValueError(f"不支持的聚合方法: {self.method}")

    def aggregate(self, gradients_list):
        """
        使用选定的方法执行安全聚合
        :param gradients_list: 工作节点的梯度列表（明文形式）
        :return: 聚合后的加密结果
        """
        if self.method == "paillier":
            return self._paillier_aggregate(gradients_list)
        else:
            raise ValueError(f"不支持的聚合方法: {self.method}")

    def _paillier_aggregate(self, gradients_list):
        """
        使用Paillier同态加密执行聚合
        :param gradients_list: 需要聚合的梯度列表
        :return: 加密后的聚合结果
        """
        logger.debug("开始执行Paillier同态加密聚合...")

        # 初始化加密后的和为0
        encrypted_sum = self.public_key.encrypt(0)

        # 对每个梯度进行加密并同态相加
        for gradient in gradients_list:
            # 确保梯度是浮点数列表（不是numpy数组）
            if isinstance(gradient, np.ndarray):
                gradient = gradient.tolist()  # 将numpy数组转换为列表
            if not isinstance(gradient, list):
                gradient = [float(gradient)]  # 确保是浮点数列表

            logger.debug(f"正在加密梯度: {gradient}")
            
            # 加密梯度列表中的每个元素
            for g in gradient:
                encrypted_gradient = self.public_key.encrypt(g)  # 加密单个值
                encrypted_sum += encrypted_gradient

        logger.debug(f"加密后的聚合结果: {encrypted_sum}")
        return encrypted_sum

    def decrypt(self, encrypted_data):
        """
        解密聚合后的加密数据
        :param encrypted_data: 加密后的聚合数据
        :return: 解密后的梯度（聚合后的结果）
        """
        if self.method == "paillier":
            decrypted_result = self.private_key.decrypt(encrypted_data)
            logger.debug(f"解密后的聚合结果: {decrypted_result}")
            return decrypted_result
        else:
            raise ValueError("仅支持Paillier加密方式的解密")

    def initialize_global_weights(self, gradient_shape):
        """
        初始化全局模型权重为零（与梯度形状相同）
        :param gradient_shape: 梯度数组的形状
        :return: 初始化的全局权重（numpy数组）
        """
        logger.debug(f"正在初始化形状为 {gradient_shape} 的全局权重")
        return np.zeros(gradient_shape)