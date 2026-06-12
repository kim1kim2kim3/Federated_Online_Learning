import sys
from config import config
sys.path.append('./fl-server')
from Refol import REFOL
from Fedavg import FedAvg
from Fedostc import FedOSTC


if __name__ == "__main__":

    server_registry = {
        "refol": REFOL,  # REFOL
        "fedavg": FedAvg, # FedAvg
        "fedostc": FedOSTC,
    }
    agg_model = (config.agg_model or "").lower()
    if not agg_model:
        supported = " or ".join(f'"{name}"' for name in server_registry)
        raise ValueError(f"agg_model is empty. Set agg_model to {supported}.")
    if agg_model not in server_registry:
        supported = ", ".join(f'"{name}"' for name in server_registry)
        raise ValueError(f'Unsupported agg_model "{config.agg_model}". Set agg_model to one of: {supported}.')

    fl_server = server_registry[agg_model](config)

    fl_server.boot()
    fl_server.run()

# 나중에 다른 알고리즘(예: FedAvg, FedProx)이 추가될 때의 예시
# fl_server = {
#     "refol": REFOL(config),
#     "fedavg": FedAvg(config),
#     "fedprox": FedProx(config)
# }[config.agg_model]
