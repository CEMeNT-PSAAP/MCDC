import os
import glob

for name in glob.glob("*.png"):
    os.system("display %s" % name)
