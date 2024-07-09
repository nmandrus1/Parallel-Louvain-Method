# GPU Accelerated Community Detection on Large Scale Graphs

TODO: 
 - [ ] Migrate to AMD ROCm/HIP
 - [x] Cleanup codebase (remove dead code)
 - [x] Implement CSR for Graph Representation (with bitfield)
    - [x] Unit tests passing for original sequential tests
    - [x] Improve Memory usage for converting edge list to CSR
 - [x] Sequential Louvain Method
 - [x] Parallel Louvain Method (MPI) (in progress)
 - [ ] Parallel Louvain Method (MPI + ROCm/HIP)

## Requirements
- CMake 3.18 or higher 
- MPI 3.1 or higher
- (Optional) [GPTL](https://jmrosinski.github.io/GPTL/index.html) with GPTL environment variable pointing to installation directory

## Building

From the root directory of the project, use CMake to generate configuration

```
$ cmake .
```

To build the community detection binaries simply run `make`


## Usage

The parallel binary makes several assumptions 

- The file passed is a directory containing files named 0..M-1 where M is the number of parallel processes running
- There are as many data files as there will be processes running. 
- The data files contain vertices labeled 0..N-1 where N is the total number of vertices in the graph
- The data files are edge lists

The program makes no assumption about whether it owns any of the vertices contained in the edge list, it will sort that out during the construction of the graph.

To run the program: 
```
$ mpirun -n 4 ./build/community data/graph
```

This will start each process looking for a file in the `graph/` directory.
