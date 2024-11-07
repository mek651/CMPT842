"""quickstart_compose: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, NDArrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from typing import Dict, List, Optional, Tuple

from quickstart_compose.task import (
    Net, 
    get_weights, 
    test, 
    load_test_datasets, 
    set_weights,
    device
)

DEVICE = device();


# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:

    net = Net().to(DEVICE)
    testloader = load_test_datasets()
    set_weights(net, parameters)  
    loss, accuracy = test(net, testloader, DEVICE)

    return loss, {"accuracy": accuracy}



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        min_evaluate_clients=2,
        initial_parameters=parameters,
        evaluate_fn=evaluate,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
