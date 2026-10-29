"""CUDA DLL discovery for native Windows conda environments."""
import os
import sys
from pathlib import Path

_dll_handles = []


def prepare_cuda_environment():
    if os.name != 'nt':
        return []
    prefixes = [Path(sys.prefix)]
    conda_exe = os.environ.get('CONDA_EXE')
    if conda_exe:
        prefixes.append(Path(conda_exe).resolve().parents[1])
    if Path(sys.prefix).parent.name.lower() == 'envs':
        prefixes.append(Path(sys.prefix).parents[1])
    candidates = [prefix / 'Library' / 'bin' for prefix in prefixes]
    candidates += [Path(os.environ[key]) / 'bin' for key in ('CUDA_PATH', 'CUDA_PATH_V11_2')
                   if os.environ.get(key)]
    directories = []
    for candidate in candidates:
        if candidate not in directories and any((candidate / name).is_file() for name in (
                'cudart64_110.dll', 'cublas64_11.dll', 'cudnn64_8.dll')):
            directories.append(candidate)
    current = os.environ.get('PATH', '').split(os.pathsep)
    missing = [str(directory) for directory in directories
               if str(directory).lower() not in {item.lower() for item in current}]
    os.environ['PATH'] = os.pathsep.join(missing + current)
    if hasattr(os, 'add_dll_directory'):
        _dll_handles.extend(os.add_dll_directory(str(directory)) for directory in directories)
    return [str(directory) for directory in directories]
