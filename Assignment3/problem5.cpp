//problem5.cpp
//
//how to use
//
//g++ -Wall -O0 -o p5 problem5.cpp
//./p5

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>

static uint64_t double_to_bits(double x)
{
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static void print_double(const char *label, double x)
{
    printf("%-46s  %.20e (hex: 0x%016llX)\n", label, x, (unsigned long long)double_to_bits(x));
}

int main(void)
{
    const int N = 100000;
    const double DELTA = 1e-18;
    const int W = 60;

    printf("\n");

    printf("DELTA = %.1e, N = %d, N*DELTA = %.1e\n\n", DELTA, N, (double)N * DELTA);

    printf("sum_a = 1.0; for N iterations: sum_a += 1e-18\n");
    printf("%.*s\n", W, "============================================================");

    double sum_a = 1.0;
    for (int i = 0; i < N; i++)
        sum_a += DELTA;

    print_double("sum_a:", sum_a);
    print_double("1.0:", 1.0);
    printf("sum_a == 1.0? %s\n\n",
        (double_to_bits(sum_a) == double_to_bits(1.0)) ? "YES" : "NO");

    printf("sum_b = 0.0; for N iterations: sum_b += 1e-18; sum_b += 1.0\n");
    printf("%.*s\n", W, "============================================================");

    double sum_b = 0.0;
    for (int i = 0; i < N; i++)
        sum_b += DELTA;

    print_double("sum_b after loop:", sum_b);
    print_double("N * DELTA:", (double)N * DELTA);

    sum_b += 1.0;
    print_double("sum_b after += 1.0:", sum_b);
    printf("\n");

    printf("Now Compare\n");
    printf("%.*s\n", W, "============================================================");
    print_double("sum_a:", sum_a);
    print_double("sum_b:", sum_b);

    double diff = sum_b - sum_a;
    print_double("diff = sum_b - sum_a:", diff);
    printf("\n");

    double ideal_increment = (double)N * DELTA;
    printf("Ideal increment N*DELTA = %.6e\n", ideal_increment);
    printf("Actual diff (sum_b - sum_a) = %.6e\n", diff);
    printf("Ratio diff / (N*DELTA) = %.6f\n", diff / ideal_increment);
    printf("\n");

    printf("Bit Differences\n");
    printf("%.*s\n", W, "============================================================");

    auto print_bits = [](const char *label, double x) 
    {
        uint64_t b = 0;
        std::memcpy(&b, &x, 8);
        int    sign = (int)((b >> 63) & 1);
        int    exp  = (int)((b >> 52) & 0x7FF);
        uint64_t mantissa = b & 0x000FFFFFFFFFFFFFULL;
        printf("%-30s sign=%d exp=%4d (biased) mantissa=0x%013llX\n", label, sign, exp, (unsigned long long)mantissa);
    };

    print_bits("1.0:", 1.0);
    print_bits("sum_a:", sum_a);
    print_bits("sum_b:", sum_b);
    print_bits("diff:", diff);
    print_bits("1e-13 (N*DELTA):", ideal_increment);
    printf("\n");

    return 0;
}
