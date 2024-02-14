import os
import glob

for name in glob.glob("results/*.png"):
    os.system("display %s" % name)
