#!/usr/bin/env python3
import sys
import os
import platform
import subprocess
import shutil

# ANSI colors for beautiful terminal output
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{BOLD}{BLUE}=== {title} ==={RESET}")

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "N/A"

def check_system():
    print_header("System Information")
    print(f"{BOLD}OS:{RESET} {platform.system()} {platform.release()}")
    print(f"{BOLD}Kernel:{RESET} {platform.version()}")
    print(f"{BOLD}Architecture:{RESET} {platform.machine()}")
    
    # CPU
    cpu_model = run_cmd("lscpu | grep 'Model name' | cut -d':' -f2").strip()
    if not cpu_model:
        cpu_model = platform.processor()
    cpu_threads = os.cpu_count()
    print(f"{BOLD}CPU Model:{RESET} {cpu_model or 'Unknown'}")
    print(f"{BOLD}CPU Threads:{RESET} {cpu_threads}")
    
    # RAM
    total_ram = "Unknown"
    if shutil.which("free"):
        total_ram = run_cmd("free -h | grep Mem | awk '{print $2}'")
    print(f"{BOLD}Total RAM:{RESET} {total_ram}")
    
    # Storage
    total_disk = "Unknown"
    free_disk = "Unknown"
    if shutil.which("df"):
        total_disk = run_cmd("df -h / | tail -n 1 | awk '{print $2}'")
        free_disk = run_cmd("df -h / | tail -n 1 | awk '{print $4}'")
    print(f"{BOLD}Storage (Root):{RESET} {free_disk} available / {total_disk} total")

def check_software():
    print_header("Software Environment")
    print(f"{BOLD}Python Executable:{RESET} {sys.executable}")
    print(f"{BOLD}Python Version:{RESET} {platform.python_version()}")
    
    # pip version
    pip_ver = run_cmd("pip3 --version | cut -d' ' -f2")
    if pip_ver == "N/A":
        pip_ver = run_cmd("pip --version | cut -d' ' -f2")
    print(f"{BOLD}pip Version:{RESET} {pip_ver}")
    
    # git version
    git_ver = run_cmd("git --version | cut -d' ' -f3")
    print(f"{BOLD}Git Version:{RESET} {git_ver}")

def check_pytorch():
    print_header("PyTorch Environment")
    try:
        import torch
        import numpy
        print(f"{GREEN}✔ PyTorch successfully imported!{RESET}")
        print(f"{BOLD}PyTorch Version:{RESET} {torch.__version__}")
        print(f"{BOLD}NumPy Version:{RESET} {numpy.__version__}")
        
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            print(f"{BOLD}CUDA Available:{RESET} {GREEN}Yes{RESET}")
            print(f"{BOLD}CUDA Device Count:{RESET} {torch.cuda.device_count()}")
            print(f"{BOLD}Current CUDA Device:{RESET} {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            print(f"{BOLD}CUDA Available:{RESET} {YELLOW}No (Using CPU){RESET}")
            device = "cpu"
            
        print(f"{BOLD}Default Device:{RESET} {CYAN}{device}{RESET}")
        
        # Test basic tensor creation
        t = torch.randn(2, 3)
        print(f"{BOLD}Tensor Test:{RESET} Successfully created a {t.shape} tensor on {t.device}")
        
    except ImportError as e:
        print(f"{RED}✘ Failed to import PyTorch or NumPy: {e}{RESET}")
        print(f"{YELLOW}Please make sure you have activated the virtual environment and installed dependencies.{RESET}")

def main():
    print(f"{BOLD}{GREEN}========================================={RESET}")
    print(f"{BOLD}{GREEN}        LLM FROM SCRATCH AUDIT           {RESET}")
    print(f"{BOLD}{GREEN}========================================={RESET}")
    check_system()
    check_software()
    check_pytorch()
    print(f"{BOLD}{GREEN}========================================={RESET}\n")

if __name__ == "__main__":
    main()
