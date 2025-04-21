# gRPC Server implementation wrapper 

import grpc
from concurrent import futures
import os
import sys
import numpy as np
from collections import defaultdict
import pandas as pd
import threading


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.base_model import BaseModel
from src.utils.logging_config import get_logger
from src.utils.metrics import ModelMetrics
from src.models.logistic_regression import LogisticRegressionModel

from docker_federated.grpc.generated import federation_pb2
from docker_federated.grpc.generated import federation_pb2_grpc

logger = get_logger()

class ClientState:
    def __init__(self, client_id, model_type, data_size):
        self.client_id = client_id
        self.model_type = model_type
        self.data_size = data_size
        self.current_round = 0
        self.metrics = {}
        

def load_test_data():
    """加载测试数据集"""
    data_path = "/app/data/credit_card_test_normalized.csv"
    try:
        df = pd.read_csv(data_path)
        X = df.drop("target", axis=1).values
        y = df["target"].values
        return {"X": X, "y": y}
    except Exception as e:
        logger.error(f"加载测试数据时出错: {e}")
        return None

class FederatedLearningServicer(federation_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.test_data = load_test_data()            
        self.global_model = LogisticRegressionModel()
        self.metrics = ModelMetrics()
        self.clients = {}  # 存储客户端状态
        self.current_round = 0
        self.count = 0
        self.next_step = False
        self.client_parameters = defaultdict(dict)
        self.aggregated_parameters = None
        self.lock = threading.Lock()  
        self.expected_clients = int(os.environ.get("EXPECTED_CLIENTS", "3"))
        self.max_rounds = int(os.environ.get("MAX_ROUNDS", "10"))  # 最大训练轮次
        logger.info(f"服务器初始化完成，等待 {self.expected_clients} 个客户端注册，计划训练 {self.max_rounds} 轮")

    def aggregate_parameters(self, parameters_list):
        """聚合客户端参数"""
        try:
            logger.info(f"开始聚合参数，参数列表长度: {len(parameters_list)}")
            if not parameters_list:
                raise ValueError("参数列表为空")
                
            # 获取参数字典的结构
            param_structure = parameters_list[0]
            logger.info(f"参数结构包含 {len(param_structure)} 个键")
            aggregated = {}
            
            # 获取当前轮次参与的客户端ID
            active_client_ids = list(self.client_parameters[self.current_round].keys())
            logger.info(f"当前轮次活跃客户端IDs: {active_client_ids}")
            
            # 只计算活跃客户端的权重
            active_clients = {cid: self.clients[cid] for cid in active_client_ids if cid in self.clients}
            logger.info(f"找到 {len(active_clients)} 个有效的活跃客户端")
            total_data_size = sum(client.data_size for client in active_clients.values())
            client_weights = [client.data_size / total_data_size for client in active_clients.values()]
            logger.info(f"计算客户端权重完成: {client_weights}")
            
            # 对每个参数进行加权平均
            for i, param_name in enumerate(param_structure.keys()):
                logger.info(f"处理参数 [{i+1}/{len(param_structure)}]: {param_name}")
                # 确保所有客户端都有该参数
                if not all(param_name in params for params in parameters_list):
                    raise ValueError(f"参数 {param_name} 在某些客户端中缺失")
                
                # 获取参数的形状和类型
                param_shape = parameters_list[0][param_name].shape
                param_dtype = parameters_list[0][param_name].dtype
                logger.info(f"参数 {param_name} 形状: {param_shape}, 类型: {param_dtype}")
                
                # 初始化聚合参数
                aggregated[param_name] = np.zeros(param_shape, dtype=param_dtype)
                
                # 加权求和
                for j, (params, weight) in enumerate(zip(parameters_list, client_weights)):
                    logger.debug(f"聚合参数 {param_name}: 客户端 {j+1}/{len(parameters_list)}, 权重 {weight}")
                    param_value = params[param_name]
                    if param_value.shape != param_shape:
                        raise ValueError(f"参数 {param_name} 的形状不一致")
                    aggregated[param_name] += weight * param_value
                logger.info(f"参数 {param_name} 聚合完成")
                    
            logger.info(f"参数聚合完成，活跃客户端数: {len(active_client_ids)}")
            logger.debug(f"参数聚合详情: 客户端数量={len(parameters_list)}, 客户端权重={client_weights}")
            return aggregated
            
        except Exception as e:
            logger.error(f"参数聚合过程中发生错误: {str(e)}")
            logger.exception(e)
            raise
        
    def _serialize_parameters(self, parameters):
        """序列化模型参数"""
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
        
    def _deserialize_parameters(self, parameters):
        """反序列化模型参数"""
        deserialized = {}
        for key, value in parameters.items():
            try:
                dtype = np.dtype(value.dtype)
                arr = np.frombuffer(value.data, dtype=dtype).reshape(value.shape)
                deserialized[key] = arr[0] if len(value.shape) == 1 and value.shape[0] == 1 else arr
            except (ValueError, TypeError) as e:
                logger.error(f"Error deserializing parameter {key}: {str(e)}")
                raise ValueError(f"Failed to deserialize parameter {key}: {str(e)}")
        return deserialized
        
    def Register(self, request, context):
        """处理客户端注册请求"""
        client_id = request.client_id
        logger.info(f"接收到客户端 {client_id} 的注册请求")
        
        with self.lock:
            self.clients[client_id] = ClientState(
                client_id=client_id,
                model_type=request.model_type,
                data_size=request.data_size
            )
            logger.info(f"客户端 {client_id} 注册成功，当前 {len(self.clients)}/{self.expected_clients} 个客户端")
            if len(self.clients) >= self.expected_clients:
                self.next_step = True
            return federation_pb2.RegisterResponse(
                code=200,
                parameters=federation_pb2.ModelParameters(
                    parameters=self._serialize_parameters(self.global_model.get_parameters())
                ),
                message="注册成功"
            )
            

    def CheckTrainingStatus(self, request, context):
        client_id = request.client_id
        
        with self.lock:
            logger.info(f"客户端 {client_id} 检查训练状态，当前next_step={self.next_step}, count={self.count}")
            if self.next_step:
                code = 200
                message = "可以开始训练"
                self.count += 1
                logger.info(f"客户端 {client_id} 获得训练许可，count增加到 {self.count}/{self.expected_clients}")
                if self.count >= self.expected_clients:
                    self.next_step = False
                    self.count = 0
                    logger.info(f"所有客户端已获得训练许可，重置next_step={self.next_step}, count={self.count}")
            else:
                code = 100
                message = f"等待其他客户端, 当前{len(self.clients)}/{self.expected_clients}个客户端"
                
            return federation_pb2.TrainingStatusResponse(
                code=code,
                message=message,
                registered_clients=len(self.clients),
                total_clients=self.expected_clients
            )
            

    def SubmitUpdate(self, request, context):
        """接收客户端模型更新"""
        try:
            client_id = request.client_id
            round_num = request.round
            
            logger.info(f"接收到客户端 {client_id} 的参数更新，轮次: {round_num+1}")
            
            with self.lock:
                # 检查轮次是否匹配
                if round_num != self.current_round:
                    logger.warning(f"客户端 {client_id} 的轮次 {round_num+1} 与服务器当前轮次 {self.current_round+1} 不匹配")
                    return federation_pb2.ServerUpdate(
                        code=400,
                        current_round=self.current_round,
                        message=f"轮次不匹配，当前服务器轮次为 {self.current_round+1}"
                    )
                
                # 存储客户端参数
                parameters = self._deserialize_parameters(request.parameters_and_metrics.parameters.parameters)
                self.client_parameters[round_num][client_id] = parameters
                
                # 更新客户端状态
                if client_id in self.clients:
                    self.clients[client_id].current_round = round_num
                    self.clients[client_id].metrics = {
                        'accuracy': request.parameters_and_metrics.metrics.accuracy,
                        'precision': request.parameters_and_metrics.metrics.precision,
                        'recall': request.parameters_and_metrics.metrics.recall,
                        'f1': request.parameters_and_metrics.metrics.f1,
                        'auc_roc': request.parameters_and_metrics.metrics.auc_roc
                    }
                
                logger.info(f"存储客户端 {client_id} 参数更新，当前轮次 {round_num+1} 已提交 {len(self.client_parameters[round_num])}/{self.expected_clients}个客户端")
                submitted_clients = len(self.client_parameters[round_num])
                if submitted_clients >= self.expected_clients:
                    logger.info(f"轮次 {round_num+1} 所有客户端参数已收集完毕，开始处理轮次完成")
                    self._process_round_completion()
                    self.current_round += 1
                    self.next_step = True
                    if self.current_round >= self.max_rounds:
                        logger.info(f"达到最大轮次 {self.max_rounds}，结束训练")
                
                return federation_pb2.ServerUpdate(
                    code=200,
                    current_round=self.current_round,
                    message="",
                    total_clients=self.expected_clients
                )

        except Exception as e:
            logger.error(f"处理客户端更新时出错: {str(e)}")
            logger.exception(e)
            return federation_pb2.ServerUpdate(
                code=500,
                current_round=self.current_round,
                message=f"服务器错误: {str(e)}",
                total_clients=self.expected_clients
            )

    def GetGlobalModel(self, request, context):
        """提供当前全局模型参数"""
        try:
            client_id = request.client_id
            round_num = request.round
            model_parameters = self.global_model.get_parameters()
            logger.info(f"向客户端 {client_id} 提供第{round_num+1}轮全局模型")
            
            # 创建ModelParameters对象
            model_params = federation_pb2.ModelParameters(
                parameters=self._serialize_parameters(model_parameters)
            )
            
            # 创建空的TrainingMetrics对象
            metrics = self.global_model.evaluate_model(self.test_data['X'], self.test_data['y'])
            metrics = federation_pb2.TrainingMetrics(
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1=metrics['f1'],
                auc_roc=metrics['auc_roc']
            )
            return federation_pb2.ParametersAndMetrics(
                parameters=model_params,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"提供全局模型参数时出错: {str(e)}")
            logger.exception(e)  
            return federation_pb2.ParametersAndMetrics(
                parameters=federation_pb2.ModelParameters(),
                metrics=federation_pb2.TrainingMetrics()
            )

    def _process_round_completion(self):
        """处理轮次完成，聚合参数并更新全局模型"""
        try:
            logger.info(f"开始处理轮次 {self.current_round+1} 完成流程")
            parameters_list = list(self.client_parameters[self.current_round].values())
            
            logger.info(f"聚合轮次 {self.current_round+1} 的参数，客户端数: {len(parameters_list)}")
            try:
                aggregated_params = self.aggregate_parameters(parameters_list)
                logger.info(f"参数聚合成功，开始更新全局模型")
                self.global_model.set_parameters(aggregated_params)
                logger.info(f"全局模型参数更新完成")
            except Exception as e:
                logger.error(f"参数聚合或模型更新时出错: {str(e)}")
                logger.exception(e)
                raise
            
            try:
                logger.info(f"开始评估全局模型性能")
                metrics = self.global_model.evaluate_model(self.test_data['X'], self.test_data['y'])
                logger.info(f"全局模型在轮次 {self.current_round+1} 的性能: {metrics}")
            except Exception as e:
                logger.error(f"评估模型时出错: {str(e)}")
                logger.exception(e)
            
            logger.info(f"轮次 {self.current_round+1} 处理完成")
            
        except Exception as e:
            logger.error(f"处理轮次完成时出错: {str(e)}")
            logger.exception(e)

def serve():
    """启动gRPC服务器"""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024)
        ]
    )
    federation_pb2_grpc.add_FederatedLearningServicer_to_server(
        FederatedLearningServicer(), server
    )
    
    port = os.environ.get("GRPC_SERVER_PORT", "50051")
    server.add_insecure_port(f"[::]:{port}")
    
    logger.info(f"联邦学习服务器正在启动，监听端口: {port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve() 