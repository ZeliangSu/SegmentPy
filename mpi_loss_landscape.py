import os
import multiprocessing as mp

os.system("mpiexec -n {} python snippet.py".format(mp.cpu_count()))