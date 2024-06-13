#include <cassert>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>  // For std::pair
#include <sstream>  // For std::istringstream

#include "graph.h"
#include "community.h"

std::unordered_map<int, std::vector<int>> communities_to_map(std::vector<int>& node_to_comm_map) {
    std::unordered_map<int, std::vector<int>> map;

    for(int i = 0; i < node_to_comm_map.size(); i++) {
        map[node_to_comm_map[i]].push_back(i);
    }

    return map;
}

int main(int argc, char** argv) {
    assert(argc == 2);
    std::ifstream file(argv[1]);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << argv[1] << std::endl;
        return 1;
    }

    std::vector<std::pair<int, int>> edges;  // Vector to store the edges
    std::string line;

    while (getline(file, line)) {
        std::istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {  // Read two integers from the line
            edges.push_back({u, v});  // Add the edge to the vector
        } else {
            std::cerr << "Error reading line: " << line << std::endl;
        }
    }

    file.close();  // Close the file

    // Optionally, print the edges to verify
    // std::cout << "Read edges:" << std::endl;
    // for (const auto& edge : edges) {
    //     std::cout << "(" << edge.first << ", " << edge.second << ")" << std::endl;
    // }

    Graph g(edges);
    g.print_graph();
    Graph g2(edges, 16);
    g2.print_graph();

    Communities c(g);
    double mod = c.modularity();

    if(c.iterate()) {
        std::cout << "Louvain Finished!!\n"; 
        std::cout << "Modularity changed from : " << mod << " to " << c.modularity() << "\n";

        auto map = communities_to_map(c.node_to_comm_map);

        for(auto pair: map) {
            std::cout << "Community " << pair.first << ": ";
            for(auto node: pair.second) 
                std::cout << node << ", ";

            std::cout << "\n";
        }
    }
    else 
        std::cout << "Error!\n";

    return 0;
}
