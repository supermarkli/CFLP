# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from proto import federated_pb2 as federated__pb2

class FederatedLearningStub(object):
    """Service definition for Federated Learning
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendGradient = channel.unary_unary(
                '/federated.FederatedLearning/SendGradient',
                request_serializer=federated__pb2.Gradient.SerializeToString,
                response_deserializer=federated__pb2.GlobalWeights.FromString,
                )


class FederatedLearningServicer(object):
    """Service definition for Federated Learning
    """

    def SendGradient(self, request, context):
        """Sends gradients from a Worker Node to the Master Node and receives updated global weights
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FederatedLearningServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SendGradient': grpc.unary_unary_rpc_method_handler(
                    servicer.SendGradient,
                    request_deserializer=federated__pb2.Gradient.FromString,
                    response_serializer=federated__pb2.GlobalWeights.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'federated.FederatedLearning', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FederatedLearning(object):
    """Service definition for Federated Learning
    """

    @staticmethod
    def SendGradient(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/federated.FederatedLearning/SendGradient',
            federated__pb2.Gradient.SerializeToString,
            federated__pb2.GlobalWeights.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
