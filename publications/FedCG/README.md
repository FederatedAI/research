## Usage 
Here is one example to run this code:

```
python main.py \ 
    --algorithm="fedensemble" \
    --dataset="office" \
    --model="resnet18" \
    --distance="cos" \
    --ensemble="ce" \
    --name="experiment" \
    --diversity \
    --add_nosie
```

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `algorithm` | The training algorithm. Options: `local`, `fedavg`, `fedsplit`, `fedprox`, `fedensemble`. Default = `fedavg`. |
| `model` | The model architecture. Options: `lenet5`, `resnet18`. Default = `lenet5`. |
| `dataset`      | Dataset to use. Options: `digit`, `office`, `cifar`. Default = `digit`. |
| `distance`  | The distance between extractor and generator. This parameter takes effect only when the algorithm is fedensemble. Options: `none`, `mse`, `cos`. Default = `none`. |
| `ensemble`      | server ensemble scheme. Options: `kl`, `ce`. Default = `kl`. |
| `lr` | learning rate for clients' extractor and classifier optimzer and server's generator and classifier optimizer, default = 2e-4. |
| `weight_decay` | weight decay for clients' extractor and classifier optimzer and server's generator and classifier optimizer, default = 1e-5. |
| `diversity` | Whether minimizing generator diversity loss, default = False. |
| `add_noise` | Whether adding noise to train data, default = False. |
| `batch_size` | Batch size, default = `8`. |
| `communication_round` | Number of federated learning communication round, default = `50`. |
| `local_epoch` | Number of local training epochs, default = `20`. |
| `global_epoch` | Number of server training epochs, default = `20`. |
| `global_iter_per_epoch` | Number of iteration per epoch for server training, default = `100`. |
| `mu` | The proximal term parameter for FedProx, default = `0.01`. |
