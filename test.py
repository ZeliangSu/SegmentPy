from time import sleep


def sleep_fn(to_print):
    sleep(30)
    print(to_print)


if __name__ == '__main__':
    from mpi4py import MPI

    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    nb_process = communicator.Get_size()
    try:
        if rank == 0:
            # communicator.Abort()  # note: Abort() method can end all child proc
            # raise NotImplementedError
            pass
        else:
            sleep_fn('ha')
    except Exception as e:
        print(e)
        communicator.Abort()

