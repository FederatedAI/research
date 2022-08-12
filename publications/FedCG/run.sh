OMP_NUM_THREADS=1 python main.py --algorithm="fedgen" --dataset="office" --model="lenet5" --seed=1 --gpu=1 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedgen" --dataset="office" --model="lenet5" --seed=2 --gpu=2 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedgen" --dataset="office" --model="lenet5" --seed=3 --gpu=3 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedgen" --dataset="office" --model="lenet5" --seed=4 --gpu=4 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedgen" --dataset="office" --model="lenet5" --seed=5 --gpu=5

OMP_NUM_THREADS=1 python main.py --algorithm="fedavg" --dataset="office" --model="lenet5" --seed=1 --gpu=1 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedavg" --dataset="office" --model="lenet5" --seed=2 --gpu=2 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedavg" --dataset="office" --model="lenet5" --seed=3 --gpu=3 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedavg" --dataset="office" --model="lenet5" --seed=4 --gpu=4 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedavg" --dataset="office" --model="lenet5" --seed=5 --gpu=5

OMP_NUM_THREADS=1 python main.py --algorithm="fedsplit" --dataset="office" --model="lenet5" --seed=1 --gpu=1 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedsplit" --dataset="office" --model="lenet5" --seed=2 --gpu=2 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedsplit" --dataset="office" --model="lenet5" --seed=3 --gpu=3 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedsplit" --dataset="office" --model="lenet5" --seed=4 --gpu=4 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedsplit" --dataset="office" --model="lenet5" --seed=5 --gpu=5

OMP_NUM_THREADS=1 python main.py --algorithm="fedprox" --dataset="office" --model="lenet5" --seed=1 --gpu=1 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedprox" --dataset="office" --model="lenet5" --seed=2 --gpu=2 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedprox" --dataset="office" --model="lenet5" --seed=3 --gpu=3 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedprox" --dataset="office" --model="lenet5" --seed=4 --gpu=4 &
OMP_NUM_THREADS=1 python main.py --algorithm="fedprox" --dataset="office" --model="lenet5" --seed=5 --gpu=5

OMP_NUM_THREADS=1 python main.py --algorithm="local" --dataset="office" --model="lenet5" --seed=1 --gpu=1 &
OMP_NUM_THREADS=1 python main.py --algorithm="local" --dataset="office" --model="lenet5" --seed=2 --gpu=2 &
OMP_NUM_THREADS=1 python main.py --algorithm="local" --dataset="office" --model="lenet5" --seed=3 --gpu=3 &
OMP_NUM_THREADS=1 python main.py --algorithm="local" --dataset="office" --model="lenet5" --seed=4 --gpu=4 &
OMP_NUM_THREADS=1 python main.py --algorithm="local" --dataset="office" --model="lenet5" --seed=5 --gpu=5





