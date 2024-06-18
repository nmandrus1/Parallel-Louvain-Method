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
    std::string fname(argv[1]);
    Graph g(fname, false); 
    // g.from_kronecker(5, 4, 0);

    for(int i = 0; i < 2; i++) {
        
        g.print_graph();

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

        std::cout << "Making new graph from communities...\n";

        g = c.into_new_graph();
    }

    g.print_graph();


    return 0;
}
