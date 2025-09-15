#include <cstdint> 
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

struct Row {
    int     n;      // input size
    int     L;      // hidden-layer count
    int     H;      // hidden size
    int64_t flops;  // forward-pass FLOPS (64-bit to avoid overflow)
};

static inline int64_t flops(int n, int L, int H)
{
    int64_t w_in  = int64_t(n)       * H;      
    int64_t w_hid = int64_t(L - 1)   * H * H;  
    int64_t w_out = int64_t(H);                
    return 2 * (w_in + w_hid + w_out);
}

int main()
{
    //search limits
    const int MAX_N      = 150;
    const int MAX_LAYERS = 9999;
    const int MAX_H      = 9999;

    std::vector<Row> T1, T2, T3, T4;

    for (int n = 7; n <= MAX_N; ++n)
        for (int L = 1; L <= MAX_LAYERS; ++L)
            for (int H = 2; H <= MAX_H; ++H) {

                int64_t f = flops(n, L, H);
                if (f < 0) continue;           

                if (f < 600) {                               
                    T1.push_back({n, L, H, f});
                } else if (n % 4 == 0 && f < 720) {          
                    T2.push_back({n, L, H, f});
                } else if ((L % 8 == 0 || L % 16 == 0) &&    
                           f < 810) {
                    T3.push_back({n, L, H, f});
                } else if (n % 4 == 0 &&
                           (H % 8 == 0 || H % 16 == 0) &&
                           f < 1000) {                       
                    T4.push_back({n, L, H, f});
                }
            }

    std::ofstream out("nn_flops_tables.txt");
    if (!out) {
        std::cerr << "Could not open output file.\n";
        return 1;
    }

    auto dump = [&](const std::vector<Row>& v, const std::string& title) {
        out << title << " (n  L  H  FLOPS)\n"
            << "-----------------------------------\n";
        for (const auto& r : v)
            out << std::setw(3)  << r.n     << ' '
                << std::setw(3)  << r.L     << ' '
                << std::setw(4)  << r.H     << ' '
                << std::setw(6)  << r.flops << '\n';
        out << '\n';
    };

    dump(T1, "Table 1: n>6, FLOPS<600");
    dump(T2, "Table 2: 4|n, FLOPS<720, not in T1");
    dump(T3, "Table 3: 8|L or 16|L, FLOPS<810, not in T1-T2");
    dump(T4, "Table 4: 4|n & (8|H or 16|H), FLOPS<1000, unique");

    std::cout << "Done.  "
              << (T1.size() + T2.size() + T3.size() + T4.size())
              << " rows written to nn_flops_tables.txt\n";
    return 0;
}
