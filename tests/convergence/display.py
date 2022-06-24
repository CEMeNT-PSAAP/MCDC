import os, glob

cases  = ['slab_parallel_beam', 'slab_uniform_source', 'slab_reed', 
          'inf_SHEM361', 'td_slab_azurv1']

for case in cases:
    os.chdir(r"./"+case)
    for name in glob.glob("*.png"):
        print(case,name)
        os.system("display %s"%name)
    os.chdir(r"..")
