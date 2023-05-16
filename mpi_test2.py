
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
        data = {'a': 1, 'b': 2, 'c': 3}
else:
        data = None

data = comm.bcast(data, root=0)
print(f'Rank: {rank}, data: {data}')

