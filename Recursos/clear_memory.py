import os
import gc
import sys

def clear_memory():
    gc.collect()


def main():

    clear_memory()


if __name__ == "__main__":
    main()