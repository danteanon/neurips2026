import subprocess
import os
from io import StringIO
import pandas as pd

def get_free_gpu():
    """
    Find the GPU with the most available memory.
    
    Returns:
    --------
    int:
        The index of the GPU with the most available memory.
    """
    try:
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        print(gpu_stats.decode("utf-8"))
        gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")),
                            names=['memory.used', 'memory.free'],
                            skiprows=1)
        print('GPU usage:\n{}'.format(gpu_df))
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
        print("Free memory")
        print(gpu_df['memory.free'].astype(int))
        idx = gpu_df['memory.free'].astype(int).idxmax()
        print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
        return idx
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No GPUs found or nvidia-smi not available. Using CPU.")
        return 0
