import psutil
import GPUtil

def print_ram_usage():
    ram_usage = psutil.virtual_memory()
    print("=== RAM Usage ===")
    print(f"Total RAM: {ram_usage.total / (1024 ** 3):.2f} GB")
    print(f"Available RAM: {ram_usage.available / (1024 ** 3):.2f} GB")
    print(f"Used RAM: {ram_usage.used / (1024 ** 3):.2f} GB")
    print(f"RAM Usage Percentage: {ram_usage.percent}%")
    print("=================")

def print_gpu_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        for gpu in gpus:
            print(f"=== GPU {gpu.id} ({gpu.name}) ===")
            print(f"  Load: {gpu.load * 100:.1f}%")
            print(f"  Free Memory: {gpu.memoryFree / 1024:.2f} GB")
            print(f"  Used Memory: {gpu.memoryUsed / 1024:.2f} GB")
            print(f"  Total Memory: {gpu.memoryTotal / 1024:.2f} GB")
            print(f"  Temperature: {gpu.temperature} °C")
            print("====================")
    else:
        print("No GPUs found")

def main():
    print_ram_usage()
    print_gpu_usage()

if __name__ == "__main__":
    main()
