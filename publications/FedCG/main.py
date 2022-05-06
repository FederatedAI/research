import torch
import numpy as np

from config import args, logger, device
from dataset import *
from utils import BestMeter, set_seed
import torch.multiprocessing as tmx

if args.algorithm == 'local':
    from clients.local import Client
elif args.algorithm == 'fedavg':
    from clients.fedavg import Client
    from servers.fedavg import Server
elif args.algorithm == 'fedsplit':
    from clients.fedsplit import Client
    from servers.fedsplit import Server
elif args.algorithm == 'fedprox':
    from clients.fedprox import Client
    from servers.fedprox import Server
elif args.algorithm == 'feddf':
    from clients.feddf import Client
    from servers.feddf import Server
elif args.algorithm == 'fedgen':
    from clients.fedgen import Client
    from servers.fedgen import Server
elif args.algorithm == 'fedgd':
    from clients.fedgd import Client
    from servers.fedgd import Server
elif args.algorithm == 'fedgd_w':
    from clients.fedgd_w import Client
    from servers.fedgd import Server

try:
    tmx.set_start_method("spawn")
except:
    pass


def main():
    logger.info("#" * 100)
    logger.info(str(args))
    set_seed(args.seed)

    # create dataset
    logger.info("Clients' Data Partition")
    if args.dataset == "fmnist":
        dataset = FMNIST()
    elif args.dataset == "cifar":
        dataset = Cifar()
    elif args.dataset == "digit":
        dataset = Digit()
    elif args.dataset == "office":
        dataset = Office()
    elif args.dataset == "domainnet":
        dataset = Domainnet()

    # create clients and server
    if args.algorithm == 'feddf':
        trainsets, valsets, testsets, server_dataset = dataset.data_partition()
        clients = []
        for i in range(args.n_clients):
            client = Client(i, trainsets[i], valsets[i], testsets[i])
            clients.append(client)
        server = Server(clients, server_dataset)

    elif args.algorithm == 'local':
        trainsets, valsets, testsets = dataset.data_partition()
        clients = []
        for i in range(args.n_clients):
            client = Client(i, trainsets[i], valsets[i], testsets[i])
            clients.append(client)
    
    else:
        trainsets, valsets, testsets = dataset.data_partition()
        clients = []
        for i in range(args.n_clients):
            client = Client(i, trainsets[i], valsets[i], testsets[i])
            clients.append(client)
        server = Server(clients)

    # record best val acc
    best_val_accs = [0.] * args.n_clients
    test_accs = [0.] * args.n_clients
    best_rounds = [-1] * args.n_clients
    
    # federated learning process
    current_round = 0
    while True:
        logger.info("** Communication Round:[%2d] Start! **" % current_round)

        # client train
        if args.parallel:
            client_processes = []
            pool = tmx.Pool()
            for i in range(args.n_clients):
                p = pool.apply_async(clients[i].local_train, args=(current_round,))
                client_processes.append(p)
            pool.close()
            pool.join()
            for p in client_processes:
                p.get()
        else:
            for i in range(args.n_clients):
                clients[i].local_train(current_round)

        # client test
        for i in range(args.n_clients):
            val_acc = clients[i].local_val()
            if val_acc > best_val_accs[i]:
                best_val_accs[i] = val_acc
                test_accs[i] = clients[i].local_test()
                best_rounds[i] = current_round

        # round result
        logger.info("communication round:%2d, after local train result:" % current_round)
        for i in range(args.n_clients):
            logger.info("client:%2d, test acc:%2.6f, best epoch:%2d" % (i, test_accs[i], best_rounds[i]))
        
        # early stop
        early_stop = True
        for i in range(args.n_clients):
            if current_round <= best_rounds[i] + args.early_stop_rounds:
                early_stop = False
                break
        if early_stop:
            break;

        # communication
        if args.algorithm == "local":
            pass

        elif args.algorithm == "fedsplit":
            server.receive(["classifier"])
            server.send(["classifier"])

        elif args.algorithm == "fedavg" or args.algorithm == "fedprox":
            server.receive(["extractor", "classifier"])
            server.send(["extractor", "classifier"])

        elif args.algorithm == "fedgen":
            server.receive(["generator", "classifier"])
            server.global_train()
            server.send(["generator", "classifier"])

        elif args.algorithm == "feddf":
            server.receive(["extractor", "classifier"])
            server.global_train()
            server.send(["extractor", "classifier"])
            
        elif args.algorithm == "fedgd" or args.algorithm == "fedgd_w":
            server.receive(["generator", "classifier"])
            server.global_train()
            server.send(["generator", "classifier"])
            
        current_round += 1
        if current_round >= 200:
            break

    # final result
    logger.info("** Federated Learning Finish! **")
    for i in range(args.n_clients):
        logger.info("client:%2d, test acc:%2.6f, best epoch:%2d" % (i, test_accs[i], best_rounds[i]))


if __name__ == "__main__":
    main()
