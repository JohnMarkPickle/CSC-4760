//problem0.cpp
//
//How to Run
//
//gcc -Wall -o p0 problem0.cpp
//./p0 8 4 0 1
//
//The numbers above are example numbers. They can be changed
// <M> <P> <p> <i>

#include <stdio.h>
#include <stdlib.h>

//Elements owned by rank in linear distribution
int linear_local_size(int M, int P, int rank) 
{
    int base      = M / P;
    int remainder = M % P;
    return base + (rank < remainder ? 1 : 0);
}

//Global index of process rank in linear distribution
int linear_start(int M, int P, int rank) 
{
    int base      = M / P;
    int remainder = M % P;
    int extra = (rank < remainder) ? rank : remainder;
    return rank * base + extra;
}

//Global index = start of process p's block + offset i
int linear_local_to_global(int M, int P, int p, int i) 
{
    return linear_start(M, P, p) + i;
}

//global index I = i*P + p
int scatter_local_to_global(int P, int p, int i) 
{
    return i * P + p;
}

//Binary search or formula
void linear_global_to_local(int M, int P, int I, int *p_out, int *i_out) 
{
    int base      = M / P;
    int remainder = M % P;
    int p, i;

    if (remainder > 0 && I < remainder * (base + 1)) 
    {
        p = I / (base + 1);
        i = I % (base + 1);
    } 
    else 
    {
        int adjusted = I - remainder;
        p = adjusted / base;
        i = adjusted % base;
    }
    *p_out = p;
    *i_out = i;
}

void scatter_global_to_local(int P, int I, int *p_out, int *i_out) 
{
    *p_out = I % P;
    *i_out = I / P;
}

int main(int argc, char *argv[]) 
{
    if (argc != 5) 
    {
        fprintf(stderr, "Usage: %s <M> <P> <p> <i>\n", argv[0]);
        fprintf(stderr, "M : vector length\n");
        fprintf(stderr, "P : number of processes\n");
        fprintf(stderr, "p : process rank\n");
        fprintf(stderr, "i : local index\n");
        return 1;
    }

    int M = atoi(argv[1]);
    int P = atoi(argv[2]);
    int p = atoi(argv[3]);
    int i = atoi(argv[4]);

    //Validate
    if (M <= 0 || P <= 0 || P > M) 
    {
        fprintf(stderr, "Error: need 0 < P <= M.\n");
        return 1;
    }
    if (p < 0 || p >= P) 
    {
        fprintf(stderr, "Error: p must be in [0, P-1].\n");
        return 1;
    }
    int local_n = linear_local_size(M, P, p);
    if (i < 0 || i >= local_n) 
    {
        fprintf(stderr, "Error: index i = %d  out of range [0, %d) for process p = %d.\n",
                i, local_n, p);
        return 1;
    }

    //linear to global
    int I = linear_local_to_global(M, P, p, i);

    //global to scatter
    int p_prime, i_prime;
    scatter_global_to_local(P, I, &p_prime, &i_prime);

    //My masterpiece
    printf("=== Distribution summary (M = %d, P = %d) ===\n\n", M, P);
    printf("  %-8s  %-10s  %-20s  %-20s\n",
           "Process", "Lin.size", "Linear indices", "Scatter indices");
    printf("  %-8s  %-10s  %-20s  %-20s\n",
           "-------", "--------", "--------------", "---------------");

    for (int r = 0; r < P; r++) 
    {
        int sz    = linear_local_size(M, P, r);
        int start = linear_start(M, P, r);

        char lin_buf[64] = "";
        int  pos         = 0;
        for (int k = 0; k < sz; k++) 
        {
            pos += snprintf(lin_buf + pos, sizeof(lin_buf) - pos,
                            "%d%s", start + k, (k < sz - 1) ? " , " : "");
        }

        char scat_buf[64] = "";
        pos               = 0;
        int scat_sz       = M / P + (r < M % P ? 1 : 0);
        for (int k = 0; k < scat_sz; k++) 
        {
            pos += snprintf(scat_buf + pos, sizeof(scat_buf) - pos,
                            "%d%s", k * P + r, (k < scat_sz - 1) ? "," : "");
        }

        printf("  %-8d  %-10d  %-20s  %-20s\n", r, sz, lin_buf, scat_buf);
    }

    //Results
    printf("\n=== Mapping result ===\n\n");
    printf("Linear: process p = %d, local i = %d\n", p, i);
    printf("Global: I = %d\n", I);
    printf("Scatter: process p'= %d, local i' = %d\n\n", p_prime, i_prime);

    //Verify
    int I_check = scatter_local_to_global(P, p_prime, i_prime);
    if (I_check == I) 
    {
        printf("  Verification: scatter (%d,%d) -> global %d  [OK]\n",
               p_prime, i_prime, I_check);
    } 
    else 
    {
        printf("Error: got %d, expected %d\n", I_check, I);
    }

    return 0;
}
