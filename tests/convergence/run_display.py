import os, glob

# Fixed source
for task in os.scandir('./fixed_source'):
    os.chdir(task)
    for name in glob.glob("*.png"):
        print(task, name)
        os.system("display %s"%name)
    os.chdir(r"../..")

# Eigenvalue
for task in os.scandir('./eigenvalue'):
    os.chdir(task)
    for name in glob.glob("*.png"):
        print(task, name)
        os.system("display %s"%name)
    os.chdir(r"../..")
