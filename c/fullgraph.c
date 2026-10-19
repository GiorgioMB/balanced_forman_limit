/* BF curvature of every edge of a binomial RGG with n uniform points on the unit torus [0,1)^d, d in {2,3}.
   usage: fullgraph d n lam seed W NW outfile
   r = (lam/(n*omega_d))^{1/d}. Writes float32 triples (t, BF, S) for edges with index e%NW==W. */
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

static inline double pd(double a) {
    a -= floor(a + 0.5);
    return a;
}

int main(int argc, char **argv) {
    int d = atoi(argv[1]);
    int N = atoi(argv[2]);
    double lam = atof(argv[3]);
    long sd = atol(argv[4]);
    int W = atoi(argv[5]), NW = atoi(argv[6]);

    double omega_d = (d == 2) ? M_PI : 4.0 * M_PI / 3.0;
    double r = pow(lam / (N * omega_d), 1.0 / d);

    seed(1234567ULL * sd + 11);
    double *X = malloc(sizeof(double) * N * d);
    for (int i = 0; i < N * d; i++)
        X[i] = unif();

    int G = (int)floor(1.0 / r);
    if (G < 3) {
        fprintf(stderr, "r too large\n");
        return 1;
    }
    double cell_side = 1.0 / G;
    int n_cells = 1;
    for (int k = 0; k < d; k++)
        n_cells *= G;

    int *cell = malloc(4 * N), *cell_start = calloc(n_cells + 1, 4), *order = malloc(4 * N);
    for (int i = 0; i < N; i++) {
        int c = 0;
        for (int k = 0; k < d; k++) {
            int ck = (int)(X[i * d + k] / cell_side);
            if (ck >= G)
                ck = G - 1;
            c = c * G + ck;
        }
        cell[i] = c;
        cell_start[c + 1]++;
    }
    for (int c = 0; c < n_cells; c++)
        cell_start[c + 1] += cell_start[c];

    int *fill = malloc(4 * (n_cells + 1));
    for (int c = 0; c <= n_cells; c++)
        fill[c] = cell_start[c];
    for (int i = 0; i < N; i++)
        order[fill[cell[i]]++] = i;

    double *Y = malloc(sizeof(double) * N * d);
    for (int a = 0; a < N; a++)
        for (int k = 0; k < d; k++)
            Y[a * d + k] = X[order[a] * d + k];

    int WD = (N + 63) / 64;
    uint64_t *B = calloc((size_t)N * WD, 8);
    long cap = 1 << 22, ne = 0;
    int *EI = malloc(4 * cap), *EJ = malloc(4 * cap);
    float *ET = malloc(4 * cap);

    int n_neigh_cells = 1;
    for (int k = 0; k < d; k++)
        n_neigh_cells *= 3;

    for (int c = 0; c < n_cells; c++) {
        int cell_coord[3];
        int rest = c;
        for (int k = d - 1; k >= 0; k--) {
            cell_coord[k] = rest % G;
            rest /= G;
        }

        for (int m = 0; m < n_neigh_cells; m++) {
            int digits = m, c2 = 0;
            for (int k = 0; k < d; k++) {
                int off = digits % 3 - 1;
                digits /= 3;
                c2 = c2 * G + ((cell_coord[k] + off + G) % G);
            }

            for (int a = cell_start[c]; a < cell_start[c + 1]; a++)
                for (int b = cell_start[c2]; b < cell_start[c2 + 1]; b++) {
                    if (b <= a)
                        continue;

                    double d2 = 0;
                    for (int k = 0; k < d; k++) {
                        double z = pd(Y[a * d + k] - Y[b * d + k]);
                        d2 += z * z;
                    }

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

    FILE *fo = fopen(argv[7], "wb");
    uint64_t *Ui = malloc(8 * WD), *Uj = malloc(8 * WD);
    float buf[3];

    for (long e = W; e < ne; e += NW) {
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

        double bf = 0, S = 0;
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
            S = gm > 0 ? (nzk + nzw) / (gm * dmax) : 0.0;
            bf = 2.0 / di + 2.0 / dj - 2 + 2.0 * tri / dmax + tri / dmin + S;
        }

        buf[0] = ET[e];
        buf[1] = bf;
        buf[2] = S;
        fwrite(buf, 4, 3, fo);
    }

    fclose(fo);
    if (W == 0)
        fprintf(stderr, "edges %ld r %.5f\n", ne, r);
    return 0;
}
