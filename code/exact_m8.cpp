// Exhaust all C(36,8) seeds on the finite 6x6 square, for exactly three steps.
// Build: g++ -O3 -std=c++17 exact_m8.cpp -o exact_m8
// No symmetry reduction, solver, threads, or external dependencies.
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using U = std::uint64_t;
constexpr U MASK = (U(1) << 36) - 1;
constexpr U LEFT = (U(1)<<0)|(U(1)<<6)|(U(1)<<12)|(U(1)<<18)|(U(1)<<24)|(U(1)<<30);
constexpr U RIGHT = LEFT << 5;

U step(U x) {
    const U west = (x & ~RIGHT) << 1;
    const U east = (x & ~LEFT) >> 1;
    const std::array<U,8> neighbours = {
        west, east, x<<6, x>>6, west<<6, west>>6, east<<6, east>>6};
    U ones=0, twos=0, fours=0, eights=0;
    for (U a : neighbours) {
        U carry = ones & a; ones ^= a;
        U carry2 = twos & carry; twos ^= carry;
        U carry3 = fours & carry2; fours ^= carry2;
        eights ^= carry3;
    }
    return MASK & ~eights & ~fours & twos & (ones | x);
}

// Intentionally scalar, using coordinates rather than shifts/bitplane sums.
U reference(U x) {
    U y=0;
    for (int r=0; r<6; ++r) for (int c=0; c<6; ++c) {
        int count=0;
        for (int dr=-1; dr<=1; ++dr) for (int dc=-1; dc<=1; ++dc) {
            int rr=r+dr, cc=c+dc;
            if ((dr || dc) && rr>=0 && rr<6 && cc>=0 && cc<6)
                count += (x >> (rr*6+cc)) & 1;
        }
        if (count==3 || (count==2 && ((x>>(r*6+c))&1))) y |= U(1)<<(r*6+c);
    }
    return y;
}

int population(U x) { return __builtin_popcountll(x); }

void self_test() {
    std::mt19937_64 random(20261031);
    for (int i=0; i<20000; ++i) {
        U x = i<36 ? U(1)<<i : random() & MASK;
        for (int t=0; t<3; ++t) {
            U y=reference(x);
            if (step(x)!=y) throw std::runtime_error("simulator mismatch");
            x=y;
        }
    }
    if (step(0)!=0 || step(MASK)!=reference(MASK))
        throw std::runtime_error("boundary mismatch");
}

int main(int argc, char** argv) {
    try {
        if (argc==3 && std::string(argv[1])=="--step") {
            U x=std::stoull(argv[2]);
            if (x>MASK) throw std::runtime_error("board exceeds 36 bits");
            std::cout << step(x) << '\n';
            return 0;
        }
        if (argc!=1) throw std::runtime_error("usage: exact_m8 [--step INTEGER]");
        self_test();
        std::cerr << "Simulator self-test passed (60000 scalar comparisons).\n";
        auto start=std::chrono::steady_clock::now();
        U count=0, best_seed=0, x=(U(1)<<8)-1;
        int best=-1;
        while (x<=MASK) {
            int final=population(step(step(step(x))));
            ++count;
            if (final>best) { best=final; best_seed=x; }
            // Gosper's successor visits every 36-bit, weight-eight mask once.
            U low=x & (~x+1), next=x+low;
            x=next | (((next^x)>>2)/low);
        }
        if (count!=30260340) throw std::runtime_error("enumeration count mismatch");
        U final=reference(reference(reference(best_seed)));
        if (population(best_seed)!=8 || population(final)!=best)
            throw std::runtime_error("witness validation failed");
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout << "{\"n\":6,\"T\":3,\"topology\":\"s\",\"m\":8,"
                  << "\"status\":\"OPTIMAL\",\"method\":\"exhaustive-fixed-m\","
                  << "\"boards_scanned\":" << count << ",\"final_cells\":" << best
                  << ",\"seed\":" << best_seed << ",\"seconds\":" << seconds << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
