import grpc
from concurrent import futures
import numpy as np
from proto import federated_pb2, federated_pb2_grpc


class MasterNode(federated_pb2_grpc.FederatedLearningServicer):
    """
    Central Master Node for Federated Learning.
    Responsible for aggregating gradients and updating global model weights.
    """
    def __init__(self):
        self.global_weights = None  # Global model weights, initialized as None
        self.learning_rate = 0.01  # Learning rate for model updates
        print("Master Node initialized.")

    def SendGradient(self, request, context):
        """
        gRPC endpoint to receive gradients from worker nodes and return updated global weights.
        """
        gradient = np.array(request.gradient)
        print(f"Received gradient: {gradient}")

        # Initialize global weights if this is the first round
        if self.global_weights is None:
            self.global_weights = np.zeros_like(gradient)
            print("Initialized global weights.")

        # Update global weights using gradient descent
        self.global_weights -= self.learning_rate * gradient
        print(f"Updated global weights: {self.global_weights}")

        # Return the updated global weights to the worker
        return federated_pb2.GlobalWeights(weights=self.global_weights.tolist())


def start_server():
    """
    Start the gRPC server for the Master Node.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))  # Multi-threaded gRPC server
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(MasterNode(), server)
    server.add_insecure_port('[::]:50051')  # Listen on port 50051
    print("Master Node gRPC server running on port 50051...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    start_server()
