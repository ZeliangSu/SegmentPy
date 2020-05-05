from time import sleep


def sleep_fn(to_print):
    sleep(10)
    print(to_print)


if __name__ == '__main__':
    sleep_fn('ha')