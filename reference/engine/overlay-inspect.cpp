#include "memory-overlay.h"
#include <iostream>
int main(int argc,char **argv) {
    try {
        auto overlay=flash_memory::overlay::read();
        if (!overlay) throw std::runtime_error("No overlay supplied");
        if (argc>1) overlay->verify_model(std::vector<std::string>(argv+1,argv+argc));
        std::cout << "accepted " << overlay->rows.size() << " rows\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << "\n"; return 1; }
}
