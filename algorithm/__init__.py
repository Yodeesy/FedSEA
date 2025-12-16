# algorithm/__init__.py
import os
import importlib


def get_all_algorithms():
    """
    Scans the 'algorithm' directory and returns a list of available algorithm names.
    Ignores __init__.py and non-python files.
    """
    return [
        model.split('.')[0]
        for model in os.listdir('algorithm')
        if not model.startswith('__') and model.endswith('.py')
    ]


# Dictionaries to store Server and Client classes for each algorithm
fed_servers = {}
fed_clients = {}

# Dynamically import all algorithms and register their Server/Client classes
for algorithm in get_all_algorithms():
    try:
        mod = importlib.import_module('algorithm.' + algorithm)

        # Expecting class names to follow the convention: <AlgoName>Server / <AlgoName>Client
        server_class_name = algorithm + 'Server'
        client_class_name = algorithm + 'Client'

        # Register classes if they exist in the module
        if hasattr(mod, server_class_name):
            fed_servers[algorithm] = getattr(mod, server_class_name)

        if hasattr(mod, client_class_name):
            fed_clients[algorithm] = getattr(mod, client_class_name)

    except Exception as e:
        print(f"[Warning] Failed to import algorithm '{algorithm}': {e}")


def get_server(algorithm_name, args, clients, model, data, logger):
    """
    Factory method to instantiate the Server for the specified algorithm.
    """
    if algorithm_name not in fed_servers:
        raise ValueError(f"Algorithm '{algorithm_name}' not found. Available: {list(fed_servers.keys())}")
    return fed_servers[algorithm_name](args, clients, model, data, logger)


def get_client(algorithm_name, args, model, data):
    """
    Factory method to instantiate a Client for the specified algorithm.
    """
    if algorithm_name not in fed_clients:
        raise ValueError(f"Algorithm '{algorithm_name}' not found. Available: {list(fed_clients.keys())}")
    return fed_clients[algorithm_name](args, model, data)