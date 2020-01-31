import os
# prevent using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level


# os.system("mpiexec -n {} python snippet.py".format(mp.cpu_count()))
# os.system("mpiexec --use-hwthread-cpus python snippet.py")
os.system("mpiexec --use-hwthread-cpus python landscape.py")