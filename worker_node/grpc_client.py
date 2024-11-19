import grpc
from proto import federated_pb2, federated_pb2_grpc

class GRPCClient:
    """
    gRPC Client for Worker Node to communicate with Master Node.
    """

    def __init__(self, master_address="localhost:50051"):
        """
        Initialize the gRPC client with the Master Node address.
        :param master_address: The address of the Master Node (default: localhost:50051).
        """
        self.master_address = master_address
        print(f"gRPC Client initialized. Connecting to Master Node at {master_address}")

    def send_gradient(self, gradient):
        """
        Send gradient to the Master Node and receive updated global weights.
        :param gradient: The gradient array to send.
        :return: Updated global weights received from Master Node.
        """
        try:
            with grpc.insecure_channel(self.master_address) as channel:
                stub = federated_pb2_grpc.FederatedLearningStub(channel)

                # Create and send the gradient request
                request = federated_pb2.Gradient(gradient=gradient.tolist())
                print(f"Sending gradient to Master Node: {gradient}")
                response = stub.SendGradient(request)

                # Receive the updated global weights
                updated_weights = response.weights
                print(f"Received updated global weights: {updated_weights}")
                return updated_weights

        except grpc.RpcError as e:
            print(f"gRPC error: {e}")
            raise


if __name__ == "__main__":
    # Example usage of GRPCClient
    example_gradient = [0.1, 0.2, 0.3]  # Example gradient to send
    master_address = "localhost:50051"  # Address of the Master Node

    # Initialize the gRPC client and communicate with the Master Node
    client = GRPCClient(master_address=master_address)
    updated_weights = client.send_gradient(example_gradient)
    print("Updated Weights:", updated_weights)
