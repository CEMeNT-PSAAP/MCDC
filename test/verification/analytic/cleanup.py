import numpy as np
import os
import sys
            
os.system("rm *png")

# Fixed source
for task in os.scandir('./fixed_source'):
    os.chdir(task)
    os.system("rm output*")
    os.chdir(r"../..")

# Eigenvalue
'''
for task in os.scandir('./eigenvalue'):
    os.chdir(task)
    os.system("rm output*")
    os.chdir(r"../..")
'''
