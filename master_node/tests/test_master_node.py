import unittest
import grpc
from concurrent import futures
import numpy as np
import logging

# 导入生成的gRPC模块和服务器实现
from proto import federated_pb2, federated_pb2_grpc
from master_node.master_node import FederatedLearningService


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


class MockWorkerNode:
    """
    模拟工作节点客户端，用于测试主节点的gRPC服务器。
    """
    def __init__(self, server_address="localhost:50051"):
        """
        初始化模拟工作节点
        :param server_address: 服务器地址，默认为localhost:50051
        """
        self.server_address = server_address

    def send_gradient(self, gradient):
        """
        向主节点发送梯度并接收更新后的全局权重
        :param gradient: 要发送的梯度数组
        :return: 从主节点接收到的更新后的全局权重
        """
        logger.debug(f"模拟工作节点正在发送梯度: {gradient}")
        with grpc.insecure_channel(self.server_address) as channel:
            stub = federated_pb2_grpc.FederatedLearningStub(channel)
            request = federated_pb2.Gradient(gradient=gradient)
            response = stub.SendGradient(request)
            logger.debug(f"模拟工作节点收到更新后的全局权重: {response.weights}")
            return response.weights


class TestMasterNode(unittest.TestCase):
    """
    主节点的测试套件
    """
    @classmethod
    def setUpClass(cls):
        """
        测试类的设置：启动gRPC服务器
        在所有测试方法执行之前运行一次
        """
        logger.info("正在启动测试用gRPC服务器...")
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        cls.federated_service = FederatedLearningService(aggregation_method="paillier", learning_rate=0.01)
        federated_pb2_grpc.add_FederatedLearningServicer_to_server(cls.federated_service, cls.server)
        cls.server.add_insecure_port('[::]:50051')
        cls.server.start()
        logger.info("测试用gRPC服务器已在50051端口启动")

    @classmethod
    def tearDownClass(cls):
        """
        测试类的清理：停止gRPC服务器
        在所有测试方法执行完后运行一次
        """
        logger.info("正在停止测试用gRPC服务器...")
        cls.server.stop(0)
        logger.info("测试用gRPC服务器已停止")

    def setUp(self):
        """
        每个测试方法执行前的设置
        重置服务状态，确保每个测试都从相同的初始状态开始
        """
        self.__class__.federated_service.reset()

    def test_send_gradient(self):
        """
        测试从工作节点发送单个梯度并接收更新后的权重
        验证权重更新是否正确
        """
        logger.info("开始测试单个梯度发送...")
        mock_worker = MockWorkerNode()
        gradient = [0.1, 0.2, 0.3]
        updated_weights = mock_worker.send_gradient(gradient)

        # 检查更新后的权重是否正确
        # 假设初始权重为0，学习率为0.01，则期望的更新后权重应为：梯度 * -学习率
        expected_weights = [-0.001, -0.002, -0.003]
        logger.info(f"期望的权重: {expected_weights}")
        
        np.testing.assert_almost_equal(updated_weights, expected_weights, decimal=5)
        logger.info("单个梯度发送测试通过")

    def test_multiple_gradients(self):
        """
        测试连续发送多个梯度时的聚合效果
        验证最终的全局权重是否符合预期
        """
        logger.info("开始测试多个梯度的聚合...")
        mock_worker = MockWorkerNode()
        gradients = [
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.3, 0.3, 0.2]
        ]

        # 连续发送梯度并检查最终的权重
        logger.debug(f"发送梯度序列: {gradients}")
        for gradient in gradients:
            mock_worker.send_gradient(gradient)

        # 计算期望的权重
        total_gradient = np.array([0.1, 0.2, 0.3]) + np.array([0.2, 0.1, 0.4]) + np.array([0.3, 0.3, 0.2])
        expected_weights = -0.01 * total_gradient  # 学习率为0.01
        logger.info(f"期望的聚合权重: {expected_weights}")

        # 发送一个空梯度来获取最终的全局权重
        updated_weights = mock_worker.send_gradient([0, 0, 0])

        # 比较最终权重与期望值
        np.testing.assert_almost_equal(updated_weights, expected_weights, decimal=5)
        logger.info("多个梯度聚合测试通过")


if __name__ == "__main__":
    logger.info("开始运行测试套件...")
    unittest.main()
