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
from docker_federated.grpc.parameter_utils import serialize_parameters, deserialize_parameters

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
    def __init__(self, use_homomorphic_encryption=False):
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
        # 是否启用同态加密
        self.use_homomorphic_encryption = use_homomorphic_encryption or os.environ.get("USE_HOMOMORPHIC_ENCRYPTION", "False").lower() == "true"
        logger.info(f"同态加密状态: {'启用' if self.use_homomorphic_encryption else '未启用'}")
        if self.use_homomorphic_encryption:
            try:
                import pickle
                with open('/app/certs/private_key.pkl', 'rb') as f:
                    self.private_key = pickle.load(f)
                logger.info("成功加载同态加密私钥")
            except FileNotFoundError:
                logger.error("未找到同态加密私钥文件: /app/certs/private_key.pkl")
                raise
            except Exception as e:
                logger.error(f"加载同态加密私钥失败: {str(e)}")
                raise
        logger.info(f"服务器初始化完成，等待 {self.expected_clients} 个客户端注册，计划训练 {self.max_rounds} 轮")

    def aggregate_parameters(self, parameters_list):
        """聚合客户端参数"""
        try:
            logger.info(f"开始聚合参数，参数列表长度: {len(parameters_list)}")
            if not parameters_list:
                raise ValueError("参数列表为空")
                
            # 获取参数字典的结构
            param_structure = parameters_list[0]
            aggregated = {}
            
            # 获取当前轮次参与的客户端ID
            active_client_ids = list(self.client_parameters[self.current_round].keys())
            logger.info(f"当前轮次活跃客户端IDs: {active_client_ids}")
            
            # 只计算活跃客户端的权重
            active_clients = {cid: self.clients[cid] for cid in active_client_ids if cid in self.clients}
            total_data_size = sum(client.data_size for client in active_clients.values())
            client_weights = [client.data_size / total_data_size for client in active_clients.values()]
            logger.info(f"计算客户端权重完成: {client_weights}")
            
            # 对每个参数进行加权平均
            for i, param_name in enumerate(param_structure.keys()):
                if not all(param_name in params for params in parameters_list):
                    raise ValueError(f"参数 {param_name} 在某些客户端中缺失")
                
                # 获取参数的形状和类型
                param_shape = parameters_list[0][param_name].shape
                param_dtype = parameters_list[0][param_name].dtype
                
                # 初始化聚合参数
                aggregated[param_name] = np.zeros(param_shape, dtype=param_dtype)
                
                # 加权求和
                for j, (params, weight) in enumerate(zip(parameters_list, client_weights)):
                    logger.debug(f"聚合参数 {param_name}: 客户端 {j+1}/{len(parameters_list)}, 权重 {weight}")
                    param_value = params[param_name]
                    if param_value.shape != param_shape:
                        raise ValueError(f"参数 {param_name} 的形状不一致")
                    aggregated[param_name] += weight * param_value
            return aggregated
            
        except Exception as e:
            logger.error(f"参数聚合过程中发生错误: {str(e)}")
            logger.exception(e)
            raise
        
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
                    parameters=serialize_parameters(self.global_model.get_parameters())
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
                parameters = deserialize_parameters(request.parameters_and_metrics.parameters.parameters)
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
                    logger.info(f"轮次 {round_num+1} 所有客户端参数已收集完毕，开始聚合")
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

    def SubmitEncryptedUpdate(self, request, context):
        """接收客户端密文模型更新（只反序列化密文，不解密）"""
        try:
            import pickle
            client_id = request.client_id
            round_num = request.round
            logger.info(f"接收到客户端 {client_id} 的密文参数更新，轮次: {round_num+1}")
            with self.lock:
                # 检查轮次是否匹配
                if round_num != self.current_round:
                    logger.warning(f"客户端 {client_id} 的轮次 {round_num+1} 与服务器当前轮次 {self.current_round+1} 不匹配")
                    return federation_pb2.ServerUpdate(
                        code=400,
                        current_round=self.current_round,
                        message=f"轮次不匹配，当前服务器轮次为 {self.current_round+1}",
                        total_clients=self.expected_clients
                    )
                # 只反序列化密文参数，不解密
                encrypted_params = request.parameters_and_metrics.parameters.parameters
                params = {}
                for key, enc_array in encrypted_params.items():
                    flat = [pickle.loads(b) for b in enc_array.data]
                    arr = np.array(flat, dtype=object).reshape(enc_array.shape)
                    params[key] = arr
                self.client_parameters[round_num][client_id] = params
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
                logger.info(f"存储客户端 {client_id} 密文参数（未解密），当前轮次 {round_num+1} 已提交 {len(self.client_parameters[round_num])}/{self.expected_clients}个客户端")
                submitted_clients = len(self.client_parameters[round_num])
                if submitted_clients >= self.expected_clients:
                    logger.info(f"轮次 {round_num+1} 所有客户端参数已收集完毕，开始聚合")
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
            logger.error(f"处理客户端密文更新时出错: {str(e)}")
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
                parameters=serialize_parameters(model_parameters)
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

    def aggregate_encrypted_parameters(self, parameters_list, client_weights, private_key):
        """对密文参数进行同态加权聚合并解密，返回明文参数字典"""
        param_structure = parameters_list[0]
        aggregated = {}
        for param_name in param_structure.keys():
            param_shape = parameters_list[0][param_name].shape
            agg = None
            for i, (params, weight) in enumerate(zip(parameters_list, client_weights)):
                param_value = params[param_name]
                weighted = np.vectorize(lambda x: x * weight, otypes=[object])(param_value)
                if agg is None:
                    agg = weighted
                else:
                    agg = np.vectorize(lambda a, b: a + b, otypes=[object])(agg, weighted)
            decrypted = np.vectorize(private_key.decrypt)(agg)
            aggregated[param_name] = decrypted.reshape(param_shape)
        return aggregated

    def _process_round_completion(self):
        """处理轮次完成，聚合参数并更新全局模型（支持同态加密）"""
        try:
            parameters_list = list(self.client_parameters[self.current_round].values())
            if not self.use_homomorphic_encryption:
                # 明文聚合
                try:
                    aggregated_params = self.aggregate_parameters(parameters_list)
                    self.global_model.set_parameters(aggregated_params)
                    logger.info(f"全局模型参数更新完成（明文聚合）")
                except Exception as e:
                    logger.error(f"参数聚合或模型更新时出错: {str(e)}")
                    logger.exception(e)
                    raise
            else:
                # 同态加密聚合
                logger.info("开始同态加密参数聚合")
                active_client_ids = list(self.client_parameters[self.current_round].keys())
                active_clients = {cid: self.clients[cid] for cid in active_client_ids if cid in self.clients}
                total_data_size = sum(client.data_size for client in active_clients.values())
                client_weights = [client.data_size / total_data_size for client in active_clients.values()]
                logger.info(f"同态聚合权重: {client_weights}")
                aggregated = self.aggregate_encrypted_parameters(parameters_list, client_weights, self.private_key)
                self.global_model.set_parameters(aggregated)
                logger.info(f"全局模型参数更新完成（同态加密聚合+解密）")
            try:
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
    # 创建服务器
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024)
        ]
    )
    
    # 添加服务
    federation_pb2_grpc.add_FederatedLearningServicer_to_server(
        FederatedLearningServicer(use_homomorphic_encryption=True), server
    )
    
    port = os.environ.get("GRPC_SERVER_PORT", "50051")
    
    try:
        # 尝试读取服务器证书和私钥
        with open('/app/certs/server.key', 'rb') as f:
            private_key = f.read()
        with open('/app/certs/server.crt', 'rb') as f:
            certificate_chain = f.read()
            
        # 创建服务器安全凭证
        server_credentials = grpc.ssl_server_credentials(
            [(private_key, certificate_chain)]
        )
        
        # 使用安全端口
        server.add_secure_port(f"[::]:{port}", server_credentials)
        logger.info(f"联邦学习安全服务器正在启动，监听端口: {port}")
        
    except FileNotFoundError as e:
        # 如果找不到证书文件，使用不安全端口
        logger.warning(f"未找到证书文件: {str(e)}，将使用不安全端口")
        server.add_insecure_port(f"[::]:{port}")
        logger.info(f"联邦学习服务器（不安全模式）正在启动，监听端口: {port}")
        
    except Exception as e:
        # 其他错误继续抛出
        logger.error(f"服务器启动失败: {str(e)}")
        raise
    
    # 启动服务器
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve() 