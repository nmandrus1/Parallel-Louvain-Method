#include <cstdlib>
#include <mpi.h>
#include "graph.h"
#include <random>

/*
* EXAMPLE USAGE 
* graph500 <SCALE> <EDGEFACTOR> <SEED> <PARALLEL> <CUDA> <CHECKPOINT_INTERVAL> <OUTPUT>
*
*      graph500 5 16 123 0 1 2 1                 -- creates a graph using CUDA with 2^5 (32) vertices, 
*                                                   runs BFS sequentially, checkpoints every 2 iterations,
*                                                   and prints output
*
*      mpirun -n 4 graph500 5 16 123 1 0 1     -- each process creates a graph on the CPU with 2^5 (32)  
*                                                 vertices (128 total 32x4), runs BFS in parallel, 
*                                                 , never checkpoints, and prints output
*       
*      NOTES: 
*         -- A seed value of 0 will use a random seed made unique by the processes' rank
*         -- A CHECKPOINT_INTERVAL value of 0 will never checkpoint (Sequential BFS will never checkpoint)
*
* 
*/


// rng seed generation
typedef std::mt19937 rng_type;
std::uniform_int_distribution<rng_type::result_type> udist(0, 7);
rng_type rng;

struct Args {
  int scale, edgefactor, seed, checkpoint_int; 
  bool parallel, cuda, output;
};

bool parse_args(int argc, char** argv, Args* args) {
  if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <SCALE> <EDGEFACTOR> <SEED> <PARALLEL> <CUDA> <CHECKPOINT_INTERVAL> <OUTPUT>\n";
        return false;
    }

    // Parse arguments
    args->scale = std::atoi(argv[1]);
    args->edgefactor = std::atoi(argv[2]);
    args->seed = std::atoi(argv[3]);
    args->parallel = std::atoi(argv[4]) != 0;
    args->cuda= std::atoi(argv[5]) != 0;
    args->checkpoint_int = std::atoi(argv[6]);
    args->output = std::atoi(argv[7]) != 0;

  return true;
}

int main(int argc, char** argv) {

  // Init MPI
  MPI_Init(&argc, &argv);
  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  Args args;
  if(!parse_args(argc, argv, &args)) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if(args.seed == 0) {
    rng.seed(time(0));
    args.seed = udist(rng) * rank;
  }

  double start = MPI_Wtime();

  Graph g;
  
  if(args.cuda) {
    g.from_kronecker_cuda(args.scale, args.edgefactor, args.seed);
  } else {
    g.from_kronecker(args.scale, args.edgefactor, args.seed);
  }

  if(args.output) 
    g.toggle_output();

  std::vector<int> parents;
  if(args.parallel) {
    parents = g.parallel_top_down_bfs(0, args.checkpoint_int);
  } else {
    parents = g.top_down_bfs(0);
  }
  

  double end = MPI_Wtime();

  double total_time = end - start;
  if(args.output) { 
    // print timings
    for(int i = 0; i < comm_size; i++) {
      if(rank == i) {
        std::cout << "Rank " << rank << ": Total Time " << total_time << "\n"; 
        g.print_timings();
        std::cout << std::endl;
      }

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
}


