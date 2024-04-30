import subprocess
import sys
import os

# path to python.exe
python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')

# upgrade pip
subprocess.call([python_exe, "-m", "ensurepip"])
subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

# install required packages
# subprocess.call([python_exe, "-m", "pip", "uninstall", "ergo3d", "-y"])
subprocess.call([python_exe, "-m", "pip", "install", "ergo3d"])
# subprocess.call([python_exe, "-m", "pip", "install", "-U", "ergo3d"])