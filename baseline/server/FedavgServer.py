from BaseServer import BaseServer
from utils.util import get_dataset
from utils.options import args_parser
import numpy as np
import torch
import os
import random

class FedavgServer(BaseServer):
    def __init__(self, args, dataset_train, dataset_test, dict_users) -> None:
        super().__init__(args, dataset_train, dataset_test, dict_users)

if __name__ == "__main__":
    args = args_parser()
    print(args)
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    dataset_train, dataset_test, dict_users = get_dataset(args)
    server = FedavgServer(args=args, dataset_train=dataset_train, dataset_test=dataset_test, dict_users= dict_users)
    server.train()


# python -u server/FedavgServer.py  --dataset mnist  --num_channels 1 --model cnn --epochs 200 --gpu 0 --num_users 100 --flmethod fedavg > log_mnist_fedavg.txt 2>&1 &
