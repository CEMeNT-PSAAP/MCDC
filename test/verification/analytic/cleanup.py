import numpy as np
import os
from task import task


os.system("rm results/*png")
for name in task.keys():
    os.chdir(name)
    os.system("rm output*")
    os.chdir(r"..")
