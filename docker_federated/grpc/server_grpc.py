# gRPC Server implementation wrapper 

import grpc
from concurrent import futures
import logging
import os
import sys
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.federation.fed_server import FederatedServer
from src.models.base_model import BaseModel
from src.utils.logging_config import get_logger
from src.models.logistic_regression import LogisticRegressionModel

from docker_federated.grpc.generated import federation_pb2
from docker_federated.grpc.generated import federation_pb2_grpc

logger = get_logger()

class FederatedLearningServicer(federation_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.server = FederatedServer()
        self.clients = {}  # 存储客户端信息
        
    def _serialize_parameters(self, parameters):
        """序列化模型参数"""
        serialized = {}
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tobytes()
            else:
                serialized[key] = str(value).encode()
        return serialized
        
    def _deserialize_parameters(self, parameters):
        """反序列化模型参数"""
        deserialized = {}
        for key, value in parameters.items():
            try:
                # 尝试将字节转换回 numpy 数组
                deserialized[key] = np.frombuffer(value).reshape(-1)
            except:
                # 如果失败，尝试转换回原始类型
                deserialized[key] = value.decode()
        return deserialized
        
    def RegisterClient(self, request, context):
        """处理客户端注册请求"""
        try:
            client_id = request.client_id
            model_type = request.model_type
            data_size = request.data_size
            
            # 创建客户端模型
            model = LogisticRegressionModel()
            
            # 添加到服务器
            self.server.add_client(model)
            self.clients[client_id] = {
                "model": model,
                "data_size": data_size
            }
            
            logger.info(f"Client {client_id} registered successfully")
            return federation_pb2.RegisterResponse(
                success=True,
                message="Client registered successfully"
            )
            
        except Exception as e:
            logger.error(f"Error registering client: {str(e)}")
            return federation_pb2.RegisterResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
            
    def SubmitParameters(self, request, context):
        """处理参数提交请求"""
        try:
            client_id = request.client_id
            parameters = request.parameters.parameters
            metrics = request.metrics
            
            # 反序列化并更新客户端模型参数
            if client_id in self.clients:
                deserialized_params = self._deserialize_parameters(parameters)
                self.clients[client_id]["model"].set_parameters(deserialized_params)
                
                # 更新指标
                self.server.metrics.update({
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "auc_roc": metrics.auc_roc
                })
                
                logger.info(f"Parameters submitted by client {client_id}")
                return federation_pb2.ParameterResponse(
                    success=True,
                    message="Parameters submitted successfully"
                )
            else:
                raise ValueError(f"Client {client_id} not found")
                
        except Exception as e:
            logger.error(f"Error submitting parameters: {str(e)}")
            return federation_pb2.ParameterResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
            
    def GetGlobalParameters(self, request, context):
        """处理获取全局参数请求"""
        try:
            client_id = request.client_id
            
            if client_id in self.clients:
                # 获取所有客户端的参数
                client_parameters = [
                    client["model"].get_parameters()
                    for client in self.clients.values()
                ]
                
                # 聚合参数
                global_parameters = self.server.aggregate_parameters(client_parameters)
                
                # 序列化参数
                serialized_params = self._serialize_parameters(global_parameters)
                
                # 转换为protobuf消息
                parameters = federation_pb2.ModelParameters(parameters=serialized_params)
                    
                logger.info(f"Global parameters sent to client {client_id}")
                return parameters
            else:
                raise ValueError(f"Client {client_id} not found")
                
        except Exception as e:
            logger.error(f"Error getting global parameters: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return federation_pb2.ModelParameters()
            
    def SubmitMetrics(self, request, context):
        """处理指标提交请求"""
        try:
            client_id = request.client_id
            metrics = request.metrics
            
            if client_id in self.clients:
                # 更新指标
                self.server.metrics.update({
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "auc_roc": metrics.auc_roc
                })
                
                logger.info(f"Metrics submitted by client {client_id}")
                return federation_pb2.MetricsResponse(
                    success=True,
                    message="Metrics submitted successfully"
                )
            else:
                raise ValueError(f"Client {client_id} not found")
                
        except Exception as e:
            logger.error(f"Error submitting metrics: {str(e)}")
            return federation_pb2.MetricsResponse(
                success=False,
                message=f"Error: {str(e)}"
            )

def serve():
    """启动gRPC服务器"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federation_pb2_grpc.add_FederatedLearningServicer_to_server(
        FederatedLearningServicer(), server
    )
    
    port = os.environ.get("GRPC_SERVER_PORT", "50051")
    server.add_insecure_port(f"[::]:{port}")
    
    logger.info(f"Starting gRPC server on port {port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve() 