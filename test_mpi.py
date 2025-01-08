from mpi4py import MPI


print(">>>>>>> MPI Test <<<<<<<<")
print(MPI.COMM_WORLD.Get_size())
print("")
