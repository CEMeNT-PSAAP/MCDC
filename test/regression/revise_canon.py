import os


test_dirs = ["fixed_source", "eigenvalue"]

for dir in test_dirs:
    for task in os.scandir("./" + dir):
        print(task)
        os.chdir(task)
        os.system("pytest")
        os.system("mv output.h5 answer.h5")
        os.chdir(r"../..")
