import grpc
from concurrent import futures
import numpy as np
import logging
from proto import federated_pb2, federated_pb2_grpc
from master_node.aggregation import Aggregator


logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("federated_learning.log")  
    ]
)

class FederatedLearningService(federated_pb2_grpc.FederatedLearningServicer):
    """
    Federated Learning gRPC Service Implementation for Master Node.
    Handles receiving gradients from Worker Nodes and returning global weights.
    """

    def __init__(self, aggregation_method="secure", learning_rate=0.01):
        """
        Initialize the service with an aggregator and learning rate.
        :param aggregation_method: Aggregation strategy to be used (e.g., "mean", "weighted", "secure").
        :param learning_rate: Learning rate for weight updates.
        """
        # Initialize aggregator with the specified method
        self.aggregator = Aggregator(method=aggregation_method)
        self.global_weights = None
        self.learning_rate = learning_rate
        logging.info(f"Federated Learning Service initialized with aggregation method '{aggregation_method}' and learning rate {learning_rate}.")

    def SendGradient(self, request, context):
        """
        Handle the gRPC request to send gradients from worker nodes.
        :param request: Gradient message containing gradients from a worker.
        :param context: gRPC context object.
        :return: GlobalWeights message containing updated global weights.
        """
        gradient = np.array(request.gradient)
        logging.info(f"Received gradient from Worker Node: {gradient}")

        # Initialize global weights if this is the first round
        if self.global_weights is None:
            self.global_weights = self.aggregator.initialize_global_weights(gradient.shape)
            logging.info(f"Initialized global weights: {self.global_weights}")

        # Aggregate gradients (this can be extended for multi-worker aggregation)
        aggregated_gradient = self.aggregator.aggregate([gradient])

        # Check if we are using encryption or secret sharing, and handle appropriately
        if self.aggregator.method == "paillier":
            # For Paillier encryption, the result is encrypted, so we need to decrypt it
            aggregated_gradient = self.aggregator.decrypt(aggregated_gradient)

        # Update global weights using gradient descent
        self.global_weights -= self.learning_rate * aggregated_gradient
        logging.info(f"Updated global weights: {self.global_weights}")

        # Return updated global weights to the worker
        return federated_pb2.GlobalWeights(weights=self.global_weights.tolist())


def start_server(aggregation_method="paillier", learning_rate=0.01, port=50051):
    """
    Start the gRPC server for the Master Node.
    :param aggregation_method: Aggregation strategy to use ("mean", "weighted", "secure").
    :param learning_rate: Learning rate for weight updates (default: 0.01).
    :param port: Port to run the server on (default: 50051).
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))  # Multi-threaded gRPC server
    federated_service = FederatedLearningService(aggregation_method=aggregation_method, learning_rate=learning_rate)
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(federated_service, server)

    # Bind the server to the specified port
    server.add_insecure_port(f"[::]:{port}")
    logging.info(f"gRPC server running on port {port}.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    # Start the server with a secure aggregation method (e.g., Paillier encryption or secret sharing)
    start_server(aggregation_method="paillier", learning_rate=0.01)
