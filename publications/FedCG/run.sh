OMP_NUM_THREADS=1 python main.py --algorithm="fedgen" --dataset="office" --model="lenet5" --seed=1 --gpu=1 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedgen" --dataset="office" --model="lenet5" --seed=2 --gpu=2 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedgen" --dataset="office" --model="lenet5" --seed=3 --gpu=3 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedgen" --dataset="office" --model="lenet5" --seed=4 --gpu=4 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedgen" --dataset="office" --model="lenet5" --seed=5 --gpu=5 --parallel

OMP_NUM_THREADS=1 python main.py --algorithm="fedavg" --dataset="office" --model="lenet5" --seed=1 --gpu=1 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedavg" --dataset="office" --model="lenet5" --seed=2 --gpu=2 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedavg" --dataset="office" --model="lenet5" --seed=3 --gpu=3 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedavg" --dataset="office" --model="lenet5" --seed=4 --gpu=4 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedavg" --dataset="office" --model="lenet5" --seed=5 --gpu=5 --parallel

OMP_NUM_THREADS=1 python main.py --algorithm="fedsplit" --dataset="office" --model="lenet5" --seed=1 --gpu=1 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedsplit" --dataset="office" --model="lenet5" --seed=2 --gpu=2 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedsplit" --dataset="office" --model="lenet5" --seed=3 --gpu=3 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedsplit" --dataset="office" --model="lenet5" --seed=4 --gpu=4 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedsplit" --dataset="office" --model="lenet5" --seed=5 --gpu=5 --parallel

OMP_NUM_THREADS=1 python main.py --algorithm="fedprox" --dataset="office" --model="lenet5" --seed=1 --gpu=1 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedprox" --dataset="office" --model="lenet5" --seed=2 --gpu=2 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedprox" --dataset="office" --model="lenet5" --seed=3 --gpu=3 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedprox" --dataset="office" --model="lenet5" --seed=4 --gpu=4 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="fedprox" --dataset="office" --model="lenet5" --seed=5 --gpu=5 --parallel

OMP_NUM_THREADS=1 python main.py --algorithm="local" --dataset="office" --model="lenet5" --seed=1 --gpu=1 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="local" --dataset="office" --model="lenet5" --seed=2 --gpu=2 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="local" --dataset="office" --model="lenet5" --seed=3 --gpu=3 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="local" --dataset="office" --model="lenet5" --seed=4 --gpu=4 --parallel &
OMP_NUM_THREADS=1 python main.py --algorithm="local" --dataset="office" --model="lenet5" --seed=5 --gpu=5 --parallel





