import grpc
import os
import sys
import uuid
import pandas as pd
import numpy as np
import time
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.base_model import BaseModel
from src.utils.logging_config import get_logger
from src.utils.metrics import ModelMetrics
from src.models.logistic_regression import LogisticRegressionModel

from docker_federated.grpc.generated import federation_pb2
from docker_federated.grpc.generated import federation_pb2_grpc
from sklearn.model_selection import train_test_split

logger = get_logger()

class FederatedLearningClient:
    def __init__(self, data=None):
        self.client_id = os.environ.get("CLIENT_ID", str(uuid.uuid4()))
        self.model = LogisticRegressionModel()
        self.metrics = ModelMetrics()
        self.current_round = 0
        self.server_host = os.environ.get("GRPC_SERVER_HOST", "localhost")
        self.server_port = os.environ.get("GRPC_SERVER_PORT", "50051")
        self._init_data(data)
        
        try:
            # 尝试读取CA证书
            with open('/app/certs/ca.crt', 'rb') as f:
                ca_cert = f.read()
            
            # 创建安全凭证
            credentials = grpc.ssl_channel_credentials(root_certificates=ca_cert)
            
            # 创建安全通道
            self.channel = grpc.secure_channel(
                f"{self.server_host}:{self.server_port}",
                credentials,
                options=[
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024)
                ]
            )
            logger.info("使用安全通道(SSL/TLS)连接服务器")
            
        except FileNotFoundError:
            # 如果找不到证书文件，使用不安全通道
            logger.warning("未找到CA证书文件: /app/certs/ca.crt，将使用不安全通道")
            self.channel = grpc.insecure_channel(
                f"{self.server_host}:{self.server_port}",
                options=[
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024)
                ]
            )
            logger.info("使用不安全通道连接服务器")
            
        except Exception as e:
            # 其他错误继续抛出
            logger.error(f"gRPC连接初始化失败: {str(e)}")
            raise
        
        # 创建存根
        self.stub = federation_pb2_grpc.FederatedLearningStub(self.channel)
        logger.info(f"客户端 {self.client_id} 初始化完成，数据集大小: {self.data_size}")

    def _init_data(self, data):
        """初始化训练和测试数据"""
        if data is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'], test_size=0.2, random_state=42
            )
            self.train_data = {'X': X_train, 'y': y_train}
            self.test_data = {'X': X_test, 'y': y_test}
            logger.debug(f"数据集划分完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        else:
            logger.warning("未提供数据，训练和测试集将为空")
            self.train_data = None
            self.test_data = None
            
        self.data_size = len(self.train_data["X"]) if self.train_data else 0
        logger.info(f"数据初始化完成，训练数据大小: {self.data_size}")

    def _serialize_parameters(self, parameters):
        """序列化模型参数"""
        try:
            serialized = {}
            param_mapping = {
                'coef_': 'weights',
                'intercept_': 'bias'
            }
            for key, value in parameters.items():
                mapped_key = param_mapping.get(key, key)
                if isinstance(value, np.ndarray):
                    numpy_array = federation_pb2.NumpyArray(
                        data=value.tobytes(),
                        shape=list(value.shape),
                        dtype=str(value.dtype)
                    )
                    serialized[mapped_key] = numpy_array
                else:
                    arr = np.array([value])
                    numpy_array = federation_pb2.NumpyArray(
                        data=arr.tobytes(),
                        shape=[1],
                        dtype=str(arr.dtype)
                    )
                    serialized[mapped_key] = numpy_array
            return serialized
        except Exception as e:
            logger.error(f"参数序列化失败: {str(e)}")
            raise
        
    def _deserialize_parameters(self, parameters):
        """反序列化模型参数"""
        deserialized = {}
        param_mapping = {
            'weights': 'coef_',
            'bias': 'intercept_'
        }
        
        for key, value in parameters.items():
            try:
                # 使用映射后的参数名
                mapped_key = param_mapping.get(key, key)
                dtype = np.dtype(value.dtype)
                arr = np.frombuffer(value.data, dtype=dtype).reshape(value.shape)
                deserialized[mapped_key] = arr[0] if len(value.shape) == 1 and value.shape[0] == 1 else arr
            except (ValueError, TypeError) as e:
                logger.error(f"客户端 {self.client_id} Error deserializing parameter {key}: {str(e)}")
                raise ValueError(f"客户端 {self.client_id} Failed to deserialize parameter {key}: {str(e)}")
        return deserialized

    def train(self, epochs=10):
        """本地训练模型"""
        if self.train_data is None:
            logger.warning(f"客户端 {self.client_id}: 没有可用的训练数据")
            return None
        try:
            if hasattr(self.model, 'epochs'):
                self.model.epochs = epochs
            elif hasattr(self.model, 'max_iter'):
                self.model.max_iter = epochs

            self.model.train_model(self.train_data['X'], self.train_data['y'])
            metrics = self.model.evaluate_model(self.test_data['X'], self.test_data['y'])
            logger.info(f"本地训练完成，评估指标: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"本地训练失败: {str(e)}")
            raise

    def _create_parameter_update_message(self, metrics):
        """创建参数更新消息"""
        parameters = self.model.get_parameters()
        
        # 创建TrainingMetrics消息
        training_metrics = federation_pb2.TrainingMetrics(
            accuracy=metrics.get('accuracy', 0.0),
            precision=metrics.get('precision', 0.0),
            recall=metrics.get('recall', 0.0),
            f1=metrics.get('f1', 0.0),
            auc_roc=metrics.get('auc_roc', 0.0)
        )
        
        # 创建ModelParameters消息
        model_parameters = federation_pb2.ModelParameters(
            parameters=self._serialize_parameters(parameters)
        )
        
        # 创建ParametersAndMetrics消息
        params_and_metrics = federation_pb2.ParametersAndMetrics(
            parameters=model_parameters,
            metrics=training_metrics
        )
        
        return federation_pb2.ClientUpdate(
            client_id=self.client_id,
            round=self.current_round,  
            parameters_and_metrics=params_and_metrics
        )

    def participate_in_training(self, n_rounds=10):
        """参与联邦学习训练"""
        register_request = federation_pb2.ClientInfo(
            client_id=self.client_id,
            model_type="logistic_regression",
            data_size=self.data_size
        )
        register_response = self.stub.Register(register_request)

        if register_response.parameters and register_response.parameters.parameters:
            initial_params = self._deserialize_parameters(register_response.parameters.parameters)
            self.model.set_parameters(initial_params)
            logger.info("已设置初始模型参数")
        
        while True:
            status_request = federation_pb2.ClientInfo(
                client_id=self.client_id
            )
            status_response = self.stub.CheckTrainingStatus(status_request)
            if status_response.code == 100:
                logger.info(f"等待其他客户端注册 ({status_response.registered_clients}/{status_response.total_clients})")
                time.sleep(0.1)  
                continue
            else :
                logger.info(f"所有客户端已就绪，开始训练")
                break
        
        while self.current_round < n_rounds:
            logger.info(f"开始第 {self.current_round + 1} 轮训练")
            metrics = self.train()
            parameter_update = self._create_parameter_update_message(metrics)
            self.stub.SubmitUpdate(parameter_update)
            logger.info(f"客户端 {self.client_id} 提交轮次 {self.current_round+1} 的参数更新")
            
            while True:
                status_request = federation_pb2.ClientInfo(
                client_id=self.client_id
                )
                status_response = self.stub.CheckTrainingStatus(status_request)
                if status_response.code == 100:
                    logger.info(f"等待其他客户端提交参数 ({status_response.registered_clients}/{status_response.total_clients})")
                    time.sleep(0.1)  
                    continue
                elif status_response.code == 200:
                    logger.info(f"所有客户端已就绪，开始请求全局模型")
                    break
                else:
                    logger.error(f"检查训练状态失败，状态码: {status_response.code}, 消息: {status_response.message}")
                    return
            
            global_model_request = federation_pb2.GetModelRequest(
                client_id=self.client_id,
                round=self.current_round
            )
            global_model_response = self.stub.GetGlobalModel(global_model_request)
            logger.info(f"客户端 {self.client_id} 请求第 {self.current_round+1} 轮的全局模型")
            self.model.set_parameters(self._deserialize_parameters(global_model_response.parameters.parameters))
            
            metrics = global_model_response.metrics
            logger.info(f"全局模型指标 - 准确率: {metrics.accuracy:.4f}, 精确率: {metrics.precision:.4f}, "
                       f"召回率: {metrics.recall:.4f}, F1分数: {metrics.f1:.4f}, AUC-ROC: {metrics.auc_roc:.4f}")
            
            logger.info(f"客户端 {self.client_id} 更新模型参数")
            self.current_round += 1
        logger.info(f"客户端 {self.client_id} 完成训练")
        
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'channel'):
                self.channel.close()
                logger.info("已关闭gRPC通道")
        except Exception as e:
            logger.error(f"关闭gRPC通道时出错: {str(e)}")

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

def main():
    client_data = load_client_data()
    if client_data:
        client = FederatedLearningClient(data=client_data)
        try:
            client.participate_in_training(n_rounds=10)
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        except Exception as e:
            logger.error(f"训练过程中发生错误: {str(e)}")
    else:
        logger.error("无法加载数据，客户端无法启动。")

if __name__ == "__main__":
    main() 
        