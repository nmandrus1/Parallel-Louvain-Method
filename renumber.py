from mpi4py import MPI
import argparse
import os
import glob

def distribute_files(files, rank, size):
    # Evenly distribute files across MPI processes
    return [files[i] for i in range(rank, len(files), size)]

def read_edge_lists(files):
    edges = []
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split())
                edges.append((src, dst))
    return edges

def gather_unique_vertices(edges):
    unique_vertices = set()
    for src, dst in edges:
        unique_vertices.update([src, dst])
    return unique_vertices

def create_global_vertex_map(comm, local_vertices):
    # Gather all unique vertices from all processes
    all_vertices = comm.allgather(local_vertices)
    global_vertices = set()
    for vertex_set in all_vertices:
        global_vertices.update(vertex_set)

    # Create global renumbering map
    vertex_map = {vertex: i for i, vertex in enumerate(sorted(global_vertices))}
    return vertex_map

def renumber_edges(edges, vertex_map):
    renumbered_edges = [(vertex_map[src], vertex_map[dst]) for src, dst in edges]
    return renumbered_edges

def write_renumbered_edges(renumbered_edges, rank, directory):
    for edges in renumbered_edges:
        new_filename = os.path.join(directory, f"{rank}")
        with open(new_filename, 'w') as f:
            for src, dst in edges:
                f.write(f"{src} {dst}\n")

def main(directory, rank, size, comm):
    all_files = glob.glob(os.path.join(directory, 'x*')) if rank == 0 else []
    all_files = comm.bcast(all_files, root=0)  # Broadcast file list to all processes

    assigned_files = distribute_files(all_files, rank, size)
    local_edges = read_edge_lists(assigned_files)
    local_vertices = gather_unique_vertices(local_edges)
    
    global_vertex_map = create_global_vertex_map(comm, local_vertices)
    renumbered_edges = renumber_edges(local_edges, global_vertex_map)
    
    # Distribute edges so each process writes its own files
    write_renumbered_edges([renumbered_edges], rank, directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel renumbering of graph edges using MPI.")
    parser.add_argument("directory", type=str, help="Directory containing edge list files")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    main(args.directory, rank, size, comm)
