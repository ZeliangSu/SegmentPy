import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# os.system("mpiexec -n 3 python inference.py")
os.system("mpiexec --use-hwthread-cpus python inference.py")