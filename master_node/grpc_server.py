import grpc
from concurrent import futures
import numpy as np
from proto import federated_pb2, federated_pb2_grpc
from aggregation import Aggregator


class FederatedLearningService(federated_pb2_grpc.FederatedLearningServicer):
    """
    Federated Learning gRPC Service Implementation for Master Node.
    Handles receiving gradients from Worker Nodes and returning global weights.
    """
    def __init__(self, aggregation_method="mean"):
        """
        Initialize the service with an aggregator.
        :param aggregation_method: Aggregation strategy to be used.
        """
        self.aggregator = Aggregator(method=aggregation_method)
        self.global_weights = None
        print("Federated Learning Service initialized.")

    def SendGradient(self, request, context):
        """
        Handle the gRPC request to send gradients from worker nodes.
        :param request: Gradient message containing gradients from a worker.
        :param context: gRPC context object.
        :return: GlobalWeights message containing updated global weights.
        """
        gradient = np.array(request.gradient)
        print(f"Received gradient from Worker Node: {gradient}")

        # Initialize global weights if this is the first round
        if self.global_weights is None:
            self.global_weights = self.aggregator.initialize_global_weights(gradient.shape)
            print(f"Initialized global weights: {self.global_weights}")

        # Aggregate gradients and update global weights
        self.global_weights -= 0.01 * gradient  # Simple SGD update
        print(f"Updated global weights: {self.global_weights}")

        # Return updated global weights to the worker
        return federated_pb2.GlobalWeights(weights=self.global_weights.tolist())


def start_server(aggregation_method="mean", port=50051):
    """
    Start the gRPC server for the Master Node.
    :param aggregation_method: Aggregation strategy to use ("mean", "weighted", "secure").
    :param port: Port to run the server on (default: 50051).
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))  # Multi-threaded gRPC server
    federated_service = FederatedLearningService(aggregation_method=aggregation_method)
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(federated_service, server)

    # Bind the server to the specified port
    server.add_insecure_port(f"[::]:{port}")
    print(f"gRPC server running on port {port}...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    # Start the server with default aggregation method
    start_server(aggregation_method="mean")
