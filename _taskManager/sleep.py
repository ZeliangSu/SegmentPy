from time import sleep
import argparse

# todo: this is a simulation
parser = argparse.ArgumentParser('A test')
parser.add_argument('-sec', '-seconds', type=int, metavar='', required=True, help='sleep X seconds')
args = parser.parse_args()

print('Start threading')
sleep(args.sec + 5)
print('End threading')


#################
#
#  control with timeout
#  https://stackoverflow.com/questions/43322201/how-to-kill-process-which-created-by-python-os-system
#
#################
# import subprocess
#
# process = subprocess.Popen(['git', 'clone', 'https://github.com/username/reponame'])
# try:
#     print('Running in process', process.pid)
#     process.wait(timeout=10)
# except subprocess.TimeoutExpired:
#     print('Timed out - killing', process.pid)
#     process.kill()
# print("Done")