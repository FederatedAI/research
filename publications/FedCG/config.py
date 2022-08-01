import argparse
import datetime
import logging
import os

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default="fedavg",
                    choices=["local", "fedavg", "fedsplit", "fedprox", "fedgen", "feddf", "fedcg", "fedcg_w"])
parser.add_argument('--dataset', type=str, default="fmnist",
                    choices=["fmnist", "digit", "office", "cifar", "domainnet"])
parser.add_argument('--model', type=str, default="lenet5", choices=["lenet5", "resnet18"])
parser.add_argument('--distance', type=str, default="mse", choices=["none", "mse", "cos"])

parser.add_argument('--seed', type=int, default=1, help='seed for initializing training (default: 1)')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
parser.add_argument('--lr', type=float, default=3e-4, help="learning rate for optimizer (default: 3e-4)")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay for optimizer (default: 1e-4)")

parser.add_argument('--noise_dim', type=int, default=100, help="the noise dim for the generator input (default: 100)")
parser.add_argument('--mu', type=float, default=0.01, help="mu for fedprox (default: 0.01)")
parser.add_argument('--n_clients', type=int, default=4, help="the number of clients")

parser.add_argument('--num_classes', type=int, default=10, help='num for classes')
parser.add_argument('--image_channel', type=int, default=3, help="channel for images")
parser.add_argument('--image_size', type=int, default=32, help='size for images')

parser.add_argument('--local_epoch', type=int, default=20,
                    help="the epochs for clients' local task training (default: 20)")
parser.add_argument('--gan_epoch', type=int, default=20,
                    help="the epochs for clients' local gan training (default: 20)")
parser.add_argument('--early_stop_rounds', type=int, default=20,
                    help="the early stop rounds for federated learning communication (default: 20)")
parser.add_argument('--global_epoch', type=int, default=20, help="the epochs for server's training (default: 20)")
parser.add_argument('--global_iter_per_epoch', type=int, default=100,
                    help="the number of iteration per epoch for server training (default: 100)")
parser.add_argument('--add_noise', dest='add_noise', action='store_true', default=False,
                    help="whether adding noise to image")
parser.add_argument('--noise_std', type=float, default=1., help="std for gaussian noise added to image(default: 1.)")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--data_dir', type=str, default="./data/", help="Data directory path")

# args = parser.parse_args()
args = parser.parse_known_args()[0]
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
# args.name = args.name + '-%s' % current_time

if args.dataset == "fmnist":
    args.image_channel = 1
    args.num_classes = 10
elif args.dataset == "cifar":
    args.num_classes = 10
elif args.dataset == "digit":
    args.n_clients = 4
    args.num_classes = 10
elif args.dataset == "office":
    args.n_clients = 4
    args.num_classes = 10
elif args.dataset == "domainnet":
    args.n_clients = 5
    args.num_classes = 10

if args.model == "lenet5":
    # extractor input size [3*32*32]
    args.image_size = 32
    # extractor ouput size [16*5*5]
    args.feature_num = 16
    args.feature_size = 5
elif args.model == "alexnet":
    # extractor input size [3*224*224]
    args.image_size = 224
    # extractor ouput size [192*13*13]
    args.feature_num = 192
    args.feature_size = 13
elif args.model == "resnet18":
    # extractor input size [3*224*224]
    args.image_size = 224
    # extractor ouput size [128*28*28]
    args.feature_num = 128
    args.feature_size = 28

if args.algorithm == "fedcg" or args.algorithm == "fedcg_w":
    args.name = args.algorithm + '_' + args.dataset + str(
        args.n_clients) + '_' + args.model + '_' + args.distance + '_' + str(args.seed)
else:
    args.name = args.algorithm + '_' + args.dataset + str(args.n_clients) + '_' + args.model + '_' + str(args.seed)
args.dir = 'experiments/' + 'bs' + str(args.batch_size) + 'lr' + str(args.lr) + 'wd' + str(args.weight_decay)
args.checkpoint_dir = os.path.join(args.dir, args.name, 'checkpoint')
os.makedirs(args.checkpoint_dir, exist_ok=True)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler(os.path.join(args.dir, args.name, 'log.txt'), mode='a')
fileHandler.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
