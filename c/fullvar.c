/* Sum of BF curvature over all edges of a Poisson RGG on the unit torus [0,1)^2.
   usage: fullvar lam n0 seed ngraphs W NW nbins
   For each graph prints: seed, then for each t-bin (t=dist/r in (0,1]) the sum of BF and the number of edges,
   restricted to edges with index e%NW==W (partial sums; combine over W). */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static uint64_t s[4];

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t nxt(void) {
    const uint64_t r = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return r;
}

static double unif(void) {
    return (nxt() >> 11) * 0x1.0p-53;
}

static void seed(uint64_t z) {
    for (int i = 0; i < 4; i++) {
        z += 0x9e3779b97f4a7c15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        s[i] = x ^ (x >> 31);
    }
}

static int poisson(double mu) {
    double u1 = unif(), u2 = unif();
    double z = sqrt(-2 * log(u1 + 1e-300)) * cos(2 * M_PI * u2);
    /* mu >= 1000 here: normal approximation with continuity correction */
    int k = (int)floor(mu + sqrt(mu) * z + 0.5);
    return k < 0 ? 0 : k;
}

static inline double pd(double a) {
    a -= floor(a + 0.5);
    return a;
}

int main(int argc, char **argv) {
    double lam = atof(argv[1]);
    double n0 = atof(argv[2]);
    long s0 = atol(argv[3]);
    int n_graphs = atoi(argv[4]), W = atoi(argv[5]), NW = atoi(argv[6]), NB = atoi(argv[7]);

    double r = sqrt(lam / (M_PI * n0));

    for (int g = 0; g < n_graphs; g++) {
        seed(7777777ULL * (s0 + g) + 3);

        int N = poisson(n0), WD = (N + 63) / 64;
        double *X = malloc(8 * N), *Y = malloc(8 * N);
        int G = (int)floor(1.0 / r);
        if (G < 3)
            G = 3;
        double cell_side = 1.0 / G;
        for (int i = 0; i < N; i++) {
            X[i] = unif();
            Y[i] = unif();
        }

        /* sort points by cell (column-major) to localise bit ranges */
        int *cell = malloc(4 * N), *cell_start = calloc(G * G + 1, 4), *order = malloc(4 * N);
        for (int i = 0; i < N; i++) {
            int cx = (int)(X[i] / cell_side), cy = (int)(Y[i] / cell_side);
            if (cx >= G)
                cx = G - 1;
            if (cy >= G)
                cy = G - 1;
            cell[i] = cx * G + cy;
            cell_start[cell[i] + 1]++;
        }
        for (int c = 0; c < G * G; c++)
            cell_start[c + 1] += cell_start[c];

        int *fill = malloc(4 * (G * G + 1));
        for (int c = 0; c <= G * G; c++)
            fill[c] = cell_start[c];
        for (int i = 0; i < N; i++)
            order[fill[cell[i]]++] = i;

        double *X_sorted = malloc(8 * N), *Y_sorted = malloc(8 * N);
        for (int k = 0; k < N; k++) {
            X_sorted[k] = X[order[k]];
            Y_sorted[k] = Y[order[k]];
        }
        free(X);
        free(Y);
        X = X_sorted;
        Y = Y_sorted;

        uint64_t *B = calloc((size_t)N * WD, 8);
        int cap = 1 << 20, ne = 0;
        int *EI = malloc(4 * cap), *EJ = malloc(4 * cap);
        float *ET = malloc(4 * cap);

        for (int cx = 0; cx < G; cx++)
            for (int cy = 0; cy < G; cy++) {
                int c = cx * G + cy;
                for (int a = cell_start[c]; a < cell_start[c + 1]; a++) {
                    for (int dx = -1; dx <= 1; dx++)
                        for (int dy = -1; dy <= 1; dy++) {
                            int c2 = ((cx + dx + G) % G) * G + ((cy + dy + G) % G);
                            for (int b = cell_start[c2]; b < cell_start[c2 + 1]; b++) {
                                if (b <= a)
                                    continue;
                                double ddx = pd(X[a] - X[b]), ddy = pd(Y[a] - Y[b]);
                                double d2 = ddx * ddx + ddy * ddy;

                                if (d2 <= r * r) {
                                    if (B[(size_t)a * WD + b / 64] & (1ULL << (b % 64)))
                                        continue;
                                    B[(size_t)a * WD + b / 64] |= 1ULL << (b % 64);
                                    B[(size_t)b * WD + a / 64] |= 1ULL << (a % 64);

                                    if (ne == cap) {
                                        cap *= 2;
                                        EI = realloc(EI, 4 * cap);
                                        EJ = realloc(EJ, 4 * cap);
                                        ET = realloc(ET, 4 * cap);
                                    }
                                    EI[ne] = a;
                                    EJ[ne] = b;
                                    ET[ne] = sqrt(d2) / r;
                                    ne++;
                                }
                            }
                        }
                }
            }

        double *SB = calloc(NB, 8);
        long *NBc = calloc(NB, 8);
        uint64_t *Ui = malloc(8 * WD), *Uj = malloc(8 * WD);

        for (int e = W; e < ne; e += NW) {
            int i = EI[e], j = EJ[e];

            int di = 0, dj = 0, tri = 0;
            for (int w = 0; w < WD; w++) {
                uint64_t a = B[(size_t)i * WD + w], b = B[(size_t)j * WD + w];
                di += __builtin_popcountll(a);
                dj += __builtin_popcountll(b);
                tri += __builtin_popcountll(a & b);
                Ui[w] = a & ~b;
                Uj[w] = b & ~a;
            }

            int bin = (int)(ET[e] * NB);
            if (bin >= NB)
                bin = NB - 1;

            double bf = 0;
            if (di > 1 && dj > 1) {
                Ui[j / 64] &= ~(1ULL << (j % 64));
                Uj[i / 64] &= ~(1ULL << (i % 64));

                int nzk = 0, nzw = 0, gm = 0;
                for (int w = 0; w < WD; w++) {
                    uint64_t m = Ui[w];
                    while (m) {
                        int b = __builtin_ctzll(m);
                        m &= m - 1;
                        int k = w * 64 + b;
                        int c = 0;
                        for (int u = 0; u < WD; u++)
                            c += __builtin_popcountll(B[(size_t)k * WD + u] & Uj[u]);
                        if (c) {
                            nzk++;
                            if (c > gm)
                                gm = c;
                        }
                    }
                }
                for (int w = 0; w < WD; w++) {
                    uint64_t m = Uj[w];
                    while (m) {
                        int b = __builtin_ctzll(m);
                        m &= m - 1;
                        int k = w * 64 + b;
                        int c = 0;
                        for (int u = 0; u < WD; u++)
                            c += __builtin_popcountll(B[(size_t)k * WD + u] & Ui[u]);
                        if (c) {
                            nzw++;
                            if (c > gm)
                                gm = c;
                        }
                    }
                }

                double dmax = di > dj ? di : dj, dmin = di < dj ? di : dj;
                double S = gm > 0 ? (nzk + nzw) / (gm * dmax) : 0.0;
                bf = 2.0 / di + 2.0 / dj - 2 + 2.0 * tri / dmax + tri / dmin + S;
            }

            SB[bin] += bf;
            NBc[bin]++;
        }

        printf("%ld", s0 + g);
        for (int b = 0; b < NB; b++)
            printf(" %.10g %ld", SB[b], NBc[b]);
        printf("\n");
        fflush(stdout);

        free(X);
        free(Y);
        free(B);
        free(EI);
        free(EJ);
        free(ET);
        free(SB);
        free(NBc);
        free(Ui);
        free(Uj);
        free(cell);
        free(cell_start);
        free(order);
        free(fill);
    }
    return 0;
}
