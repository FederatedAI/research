import numpy as np

client = 5
experiments = 5
bs = 16
lr = 0.0003
wd = 0.0001
dataset = "domainnet"
algs = ["local", "fedavg", "fedsplit", "fedprox", "fedgen", "fedcg_w", "feddf"]

for alg in algs:

    print(alg)

    fdir = 'experiments/bs' + str(bs) + 'lr' + str(lr) + 'wd' + str(wd) + '/' + alg + '_' + dataset + str(
        client) + '_lenet5_'
    if alg == 'fedcg_w':
        fdir += "mse_"
    nums = [[] for _ in range(client)]
    avg_nums = []

    for i in range(1, experiments + 1):
        fname = fdir + str(i) + '/log.txt'
        with open(fname, 'r') as f:
            lines = f.readlines()[-client:]
            sum_num = 0
            for j in range(client):
                num = float(lines[j].split(" test acc:")[1][:8])
                nums[j].append(num)
                sum_num += num
            avg_nums.append(sum_num / client)

    for j in range(client):
        print("client:%2d, acc:%.4f(%.4f)" % (j + 1, np.mean(np.array(nums[j])), np.std(np.array(nums[j]))))
    print("total average")
    print("%.4f(%.4f)" % (np.mean(np.array(avg_nums)), np.std(np.array(avg_nums))))
