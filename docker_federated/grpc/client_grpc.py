import grpc
import os
import sys
import time
import uuid
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.federation.fed_client import FederatedClient
from src.models.base_model import BaseModel
from src.utils.logging_config import get_logger
from src.models.logistic_regression import LogisticRegressionModel

from docker_federated.grpc.generated import federation_pb2
from docker_federated.grpc.generated import federation_pb2_grpc

logger = get_logger()

class FederatedLearningClient:
    def __init__(self, data=None):
        self.client_id = os.environ.get("CLIENT_ID", str(uuid.uuid4()))
        model = LogisticRegressionModel()
        self.client = FederatedClient(self.client_id, model, data)
        
        self.server_host = os.environ.get("GRPC_SERVER_HOST", "localhost")
        self.server_port = os.environ.get("GRPC_SERVER_PORT", "50051")
        
        self.channel = grpc.insecure_channel(f"{self.server_host}:{self.server_port}")
        self.stub = federation_pb2_grpc.FederatedLearningStub(self.channel)
        
        data_size = len(self.client.train_data["X"]) if self.client.train_data else 0
        logger.info(f"Client {self.client_id} initialized with data size: {data_size}")
        
    def register(self):
        """注册客户端到服务器"""
        try:
            # 准备注册请求
            request = federation_pb2.ClientInfo(
                client_id=self.client_id,
                model_type="logistic_regression",
                data_size=len(self.client.train_data["X"]) if self.client.train_data else 0
            )
            
            # 发送注册请求
            response = self.stub.RegisterClient(request)
            
            if response.success:
                logger.info(f"Client {self.client_id} registered successfully")
                return True
            else:
                logger.error(f"Failed to register client: {response.message}")
                return False
                
        except grpc.RpcError as e:
            logger.error(f"RPC error during registration: {str(e)}")
            return False
            
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

    def train_round(self, epochs=10):
        """执行一轮训练"""
        try:
            # 1. 本地训练
            metrics = self.client.train(epochs=epochs)
            
            # 2. 获取本地参数并序列化
            parameters = self.client.get_parameters()
            serialized_params = self._serialize_parameters(parameters)
            
            # 3. 准备参数提交请求
            param_request = federation_pb2.ParameterSubmission(
                client_id=self.client_id,
                parameters=federation_pb2.ModelParameters(
                    parameters=serialized_params
                ),
                metrics=federation_pb2.TrainingMetrics(
                    accuracy=metrics["accuracy"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1=metrics["f1"],
                    auc_roc=metrics["auc_roc"]
                )
            )
            
            # 4. 提交参数
            param_response = self.stub.SubmitParameters(param_request)
            if not param_response.success:
                logger.error(f"Failed to submit parameters: {param_response.message}")
                return False
                
            # 5. 获取全局参数
            global_request = federation_pb2.ParameterRequest(client_id=self.client_id)
            global_parameters = self.stub.GetGlobalParameters(global_request)
            
            # 6. 反序列化并更新本地参数
            deserialized_params = self._deserialize_parameters(global_parameters.parameters)
            self.client.set_parameters(deserialized_params)
            
            # 7. 提交最终指标
            metrics_request = federation_pb2.MetricsSubmission(
                client_id=self.client_id,
                metrics=federation_pb2.TrainingMetrics(
                    accuracy=metrics["accuracy"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1=metrics["f1"],
                    auc_roc=metrics["auc_roc"]
                )
            )
            
            metrics_response = self.stub.SubmitMetrics(metrics_request)
            if not metrics_response.success:
                logger.error(f"Failed to submit metrics: {metrics_response.message}")
                return False
                
            logger.info(f"Training round completed successfully for client {self.client_id}")
            return True
            
        except grpc.RpcError as e:
            logger.error(f"RPC error during training round: {str(e)}")
            return False
            
    def run(self, n_rounds=10, epochs_per_round=10):
        # 注册客户端
        if not self.register():
            logger.error("Failed to register client, exiting")
            return
            
        # 执行训练轮次
        for round_idx in range(n_rounds):
            logger.info(f"Client {self.client_id} starting training round {round_idx + 1}/{n_rounds}")
            
            if not self.train_round(epochs=epochs_per_round):
                logger.error(f"Training round {round_idx + 1} failed for client {self.client_id}")
                break
                
            # 等待一段时间，避免服务器过载
            time.sleep(1)
            
        logger.info(f"Training completed for client {self.client_id}")
        
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'channel'):
            self.channel.close()

def load_client_data():
    """加载客户端数据"""
    data_path = "/app/data/credit_card_train_normalized.csv"
    try:
        df = pd.read_csv(data_path)
        X = df.drop("target", axis=1).values
        y = df["target"].values
        data = {"X": X, "y": y}
        logger.info(f"成功加载客户端数据: {data_path}, 形状: X={X.shape}, y={y.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"数据文件未找到: {data_path}。请确保文件已正确挂载。")
        return None
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        return None

if __name__ == "__main__":
    client_data = load_client_data()
    if client_data:
        client = FederatedLearningClient(data=client_data)
        client.run()
    else:
        logger.error("无法加载数据，客户端无法启动。") 