import os
import gc
import sys

def clear_memory():
    gc.collect()

def restart_vscode():
    if sys.platform.startswith('win'):
        os.system('taskkill /F /IM Code.exe')  # Windows
    elif sys.platform.startswith('darwin'):
        os.system('pkill -f "Visual Studio Code"')  # macOS
    elif sys.platform.startswith('linux'):
        os.system('pkill -f "code"')  # Linux

    # Reabrir VS Code
    os.system('code .')

# Limpiar memoria
clear_memory()

# Reiniciar VS Code
restart_vscode()
