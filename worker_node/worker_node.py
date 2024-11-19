import grpc
import numpy as np
from proto import federated_pb2, federated_pb2_grpc
from local_model import LocalModel
from data_processing import load_local_data


class WorkerNode:
    """
    Worker Node in Federated Learning.
    Responsible for local training, gradient computation, and communication with the Master Node.
    """
    def __init__(self, data_path, target_column, master_address="localhost:50051"):
        """
        Initialize the Worker Node.
        :param data_path: Path to the local dataset.
        :param target_column: Name of the target column in the dataset.
        :param master_address: gRPC address of the Master Node.
        """
        self.data, self.target = load_local_data(data_path)
        self.local_model = LocalModel(self.data, self.target, target_column)
        self.master_address = master_address
        print(f"Worker Node initialized. Connecting to Master Node at {master_address}")

    def compute_and_send_gradient(self):
        """
        Compute local gradient and send it to the Master Node.
        """
        gradient = self.local_model.compute_gradient()
        print(f"Computed local gradient: {gradient}")

        with grpc.insecure_channel(self.master_address) as channel:
            stub = federated_pb2_grpc.FederatedLearningStub(channel)

            # Send gradient to the Master Node
            request = federated_pb2.Gradient(gradient=gradient.tolist())
            response = stub.SendGradient(request)
            print(f"Received updated global weights from Master Node: {response.weights}")

            # Update local model with global weights
            self.local_model.update_model(np.array(response.weights))


if __name__ == "__main__":
    # Example usage
    data_path = "local_data.csv"  # Path to the local dataset
    target_column = "target"  # Name of the target column
    master_address = "localhost:50051"  # Address of the Master Node

    # Initialize Worker Node
    worker_node = WorkerNode(data_path, target_column, master_address)

    # Compute and send gradient in a single round of communication
    worker_node.compute_and_send_gradient()
