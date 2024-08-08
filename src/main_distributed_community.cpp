#include <cstdlib>
#include <filesystem>
#include <gptl.h>
#include <mpi.h>
#include "distcommunity.h"
#include "graph.h"
#include <string>
#include <iostream>

namespace fs = std::filesystem;

struct Args {
  std::string infile;
  std::string outdir;
};

bool parse_args(int argc, char** argv, Args* args) {
  if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <INFILE> <OUTDIR>\n";
        return false;
    }

    // Parse arguments
    args->infile = std::string(argv[1]);
    args->outdir = std::string(argv[2]);

  return true;
}

int run(int rank, int comm_size, Args args) {
  fs::path base_dir(args.infile);

  // Ensure each rank accesses its corresponding file
  fs::path file_path = base_dir / std::to_string(rank);
  if (!fs::exists(file_path)) {
    std::cerr << "Error: Preprocessed data file " << file_path.string() << " does not exist for rank " 
              << rank << ". Please use the provided \"renumber.py\" script to process the data." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  Graph g(file_path.string(), true);
  DistCommunities dist_comm(g);

  // Create a directory for the current level
  fs::path outdir = fs::current_path() / args.outdir;
  std::cout << "Write Path : " << outdir << std::endl;

  if(rank == 0) {
    fs::remove_all(outdir);
    fs::create_directory(outdir);
  }

  double init_mod = dist_comm.modularity();
  dist_comm.iterate();

  dist_comm.write_communities_to_file(outdir);

  double new_mod = dist_comm.modularity();

  if (rank == 0) {
    std::cout << "Initial Modularity: " << init_mod << " and after one pass: " << new_mod << std::endl;
    std::cout << "Constructing New Graph...\n";
  }

  auto new_g = dist_comm.into_new_graph();
  // new_g.distributed_print();
  MPI_Barrier(MPI_COMM_WORLD);
  new_g.write_edges_to_file(outdir);

  return 0;
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

  #ifdef PROFILE_FNS
  auto ret = GPTLinitialize ();
  ret = GPTLstart ("total");
  #endif

  run(rank, comm_size, args);

  // if(rank == 0) {
  //   std::string f("./data/graph/graph.txt");
  //   g = Graph(f, false);
  //   g.print_graph();
  // }

  #ifdef PROFILE_FNS
  ret = GPTLstop("total");
  ret = GPTLpr(rank);
  GPTLpr_summary(MPI_COMM_WORLD);
  GPTLfinalize();
  #endif 

  MPI_Finalize();
}


