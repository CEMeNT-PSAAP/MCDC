import os


test_dirs = ["fixed_source", "eigenvalue"]

for dir in test_dirs:
    for task in os.scandir("./" + dir):
        os.chdir(task)
        os.system("mv output.h5 answer.h5")
        os.chdir(r"../..")
