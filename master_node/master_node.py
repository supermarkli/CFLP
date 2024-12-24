import grpc
from concurrent import futures
import numpy as np
import logging
from proto import federated_pb2, federated_pb2_grpc
from master_node.aggregation import Aggregator


# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    handlers=[
        logging.StreamHandler()
    ]
)

class FederatedLearningService(federated_pb2_grpc.FederatedLearningServicer):
    """
    联邦学习主节点的gRPC服务实现。
    负责接收工作节点的梯度并返回更新后的全局权重。
    """

    def __init__(self, aggregation_method="secure", learning_rate=0.01):
        """
        初始化服务
        :param aggregation_method: 聚合方法，默认为"secure"
        :param learning_rate: 学习率，默认为0.01
        """
        self.learning_rate = learning_rate
        self.global_weights = None  # 全局权重，初始为None
        self.accumulated_gradients = None  # 累积梯度，初始为None
        logging.info(f"联邦学习服务已初始化，学习率为 {learning_rate}。")

    def SendGradient(self, request, context):
        """
        处理来自工作节点的梯度发送请求
        :param request: 包含梯度数据的请求对象
        :param context: gRPC上下文
        :return: 包含更新后全局权重的响应对象
        """
        # 将接收到的梯度转换为numpy数组
        gradient = np.array(request.gradient, dtype=np.float64)
        logging.info(f"收到工作节点的梯度: {gradient}")

        # 如果是第一次接收梯度，初始化全局权重和累积梯度
        if self.global_weights is None:
            self.global_weights = np.zeros_like(gradient, dtype=np.float64)
            self.accumulated_gradients = np.zeros_like(gradient, dtype=np.float64)
            logging.info(f"初始化全局权重: {self.global_weights}")

        # 累积梯度：如果累积梯度为空，直接使用当前梯度；否则将当前梯度加到累积梯度上
        if self.accumulated_gradients is None:
            self.accumulated_gradients = gradient
        else:
            self.accumulated_gradients += gradient
        
        # 更新全局权重：使用负学习率乘以累积梯度
        self.global_weights = -self.learning_rate * self.accumulated_gradients
        logging.info(f"累积梯度: {self.accumulated_gradients}")
        logging.info(f"更新后的全局权重: {self.global_weights}")

        # 返回更新后的全局权重
        return federated_pb2.GlobalWeights(weights=self.global_weights.tolist())

    def reset(self):
        """
        重置服务状态
        将全局权重和累积梯度重置为None
        """
        self.global_weights = None
        self.accumulated_gradients = None
        logging.info("服务状态已重置")


def start_server(aggregation_method="paillier", learning_rate=0.01, port=50051):
    """
    启动主节点的gRPC服务器
    :param aggregation_method: 聚合策略，可选值："mean", "weighted", "secure"，默认为"paillier"
    :param learning_rate: 权重更新的学习率，默认为0.01
    :param port: 服务器监听的端口号，默认为50051
    """
    # 创建多线程gRPC服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # 创建联邦学习服务实例
    federated_service = FederatedLearningService(aggregation_method=aggregation_method, learning_rate=learning_rate)
    # 将服务添加到服务器
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(federated_service, server)

    # 绑定服务器到指定端口
    server.add_insecure_port(f"[::]:{port}")
    logging.info(f"gRPC服务器正在运行，端口号：{port}")
    # 启动服务器
    server.start()
    # 保持服务器运行直到被终止
    server.wait_for_termination()


if __name__ == "__main__":
    # 使用安全聚合方法（如Paillier加密或秘密共享）启动服务器
    start_server(aggregation_method="paillier", learning_rate=0.01)
