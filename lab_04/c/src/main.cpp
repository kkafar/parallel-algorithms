#include <cstdint>
#include <mpi.h>
#include <cstdio>

typedef uintmax_t Size_t;

struct Args {
  Size_t series;
  Size_t ppc;
  double a;
  double theta;
  Size_t iters;
};

// def 

int main(int argc, char * argv[]) {
  int rank, size;


	MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  std::printf("Hello world from process %d of %d\n", rank, size);

	MPI_Finalize();

	return 0;
}

