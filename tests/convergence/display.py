import os, glob

# Fixed source
for task in os.scandir('./fixed_source'):
    print(task)
    os.chdir(task)
    for name in glob.glob("*.png"):
        print(name)
        os.system("display %s"%name)
    os.chdir(r"../..")

# Eigenvalue
for task in os.scandir('./eigenvalue'):
    print(task)
    os.chdir(task)
    for name in glob.glob("*.png"):
        print(name)
        os.system("display %s"%name)
    os.chdir(r"../..")
