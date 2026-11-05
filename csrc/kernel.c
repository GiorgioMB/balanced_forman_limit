/* Second-order kernel E D^2_{x,y} F of the sum of BF curvatures, flat Palm model in the plane.

   kernel lam eps seed0 nsamples

   Unit radius, Poisson process of intensity lam/pi on the disc of radius R = 3.3 + eps around
   x = (0,0), with y = (eps, 0).  For each sample prints "seed total S J": the second difference
   F(P+x+y) - F(P+x) - F(P+y) + F(P) of the sum of BF curvatures over the edges that can change
   (those with an endpoint within 1 + eps of x), split into the 4-cycle part S and the max/min part
   J = -|tri| |1/d_i - 1/d_j| / 2.  The edge {x,y} itself, whose centred term has mean zero, is
   omitted (Section S8, Table S6). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bfcommon.h"

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: kernel lam eps seed0 nsamples\n");
        return 2;
    }
    double lam = atof(argv[1]), eps = atof(argv[2]);
    long s0 = atol(argv[3]);
    int n_samples = atoi(argv[4]);
    const double R = 3.3 + eps;

    for (int smp = 0; smp < n_samples; smp++) {
        seed_rng(1000003ULL * (uint64_t)(s0 + smp) + 17);
        int N = (int)poisson(lam * R * R), NT = N + 2;
        double *X = malloc(sizeof(double) * NT), *Y = malloc(sizeof(double) * NT);
        for (int i = 0; i < N; i++) {
            double rr = R * sqrt(unif()), th = 2 * M_PI * unif();
            X[i] = rr * cos(th);
            Y[i] = rr * sin(th);
        }
        X[N] = 0;
        Y[N] = 0;
        X[N + 1] = eps;
        Y[N + 1] = 0;
        int ix = N, iy = N + 1, W = (NT + 63) / 64;
        uint64_t *B = calloc((size_t)NT * W, 8);

        int G = (int)ceil(2 * R) + 1; /* grid of unit cells */
        int *head = malloc(sizeof(int) * G * G), *next = malloc(sizeof(int) * NT);
        for (int i = 0; i < G * G; i++)
            head[i] = -1;
        for (int i = 0; i < NT; i++) {
            int c = (int)floor(X[i] + R) * G + (int)floor(Y[i] + R);
            next[i] = head[c];
            head[c] = i;
        }
        int cap = 1 << 20, ne = 0;
        int *EI = malloc(sizeof(int) * cap), *EJ = malloc(sizeof(int) * cap);
        for (int i = 0; i < NT; i++) {
            int cx = (int)floor(X[i] + R), cy = (int)floor(Y[i] + R);
            for (int ax = cx - 1; ax <= cx + 1; ax++)
                for (int ay = cy - 1; ay <= cy + 1; ay++) {
                    if (ax < 0 || ay < 0 || ax >= G || ay >= G)
                        continue;
                    for (int j = head[ax * G + ay]; j >= 0; j = next[j]) {
                        if (j <= i)
                            continue;
                        double dx = X[i] - X[j], dy = Y[i] - Y[j];
                        if (dx * dx + dy * dy > 1.0)
                            continue;
                        B[(size_t)i * W + j / 64] |= 1ULL << (j % 64);
                        B[(size_t)j * W + i / 64] |= 1ULL << (i % 64);
                        int near_i = X[i] * X[i] + Y[i] * Y[i] <= (1 + eps) * (1 + eps);
                        int near_j = X[j] * X[j] + Y[j] * Y[j] <= (1 + eps) * (1 + eps);
                        if ((near_i || near_j) && !(i >= N && j >= N)) {
                            if (ne == cap) {
                                cap *= 2;
                                EI = realloc(EI, sizeof(int) * cap);
                                EJ = realloc(EJ, sizeof(int) * cap);
                            }
                            EI[ne] = i;
                            EJ[ne] = j;
                            ne++;
                        }
                    }
                }
        }

        uint64_t *present = malloc(8 * W), *Ni = malloc(8 * W), *Nj = malloc(8 * W), *Ui = malloc(8 * W),
                 *Uj = malloc(8 * W);
        double total = 0, S_total = 0, J_total = 0;
        for (int c = 0; c < 4; c++) { /* 0: P+x+y, 1: P+x, 2: P+y, 3: P */
            int has_x = (c == 0 || c == 1), has_y = (c == 0 || c == 2);
            double sign = (c == 0 || c == 3) ? 1 : -1;
            for (int w = 0; w < W; w++)
                present[w] = ~0ULL;
            if (!has_x)
                present[ix / 64] &= ~(1ULL << (ix % 64));
            if (!has_y)
                present[iy / 64] &= ~(1ULL << (iy % 64));
            for (int e = 0; e < ne; e++) {
                int i = EI[e], j = EJ[e];
                if (((i == ix || j == ix) && !has_x) || ((i == iy || j == iy) && !has_y))
                    continue;
                int di = 0, dj = 0, tri = 0;
                for (int w = 0; w < W; w++) {
                    Ni[w] = B[(size_t)i * W + w] & present[w];
                    Nj[w] = B[(size_t)j * W + w] & present[w];
                    di += __builtin_popcountll(Ni[w]);
                    dj += __builtin_popcountll(Nj[w]);
                    tri += __builtin_popcountll(Ni[w] & Nj[w]);
                }
                if (di <= 1 || dj <= 1)
                    continue;
                for (int w = 0; w < W; w++) {
                    Ui[w] = Ni[w] & ~Nj[w];
                    Uj[w] = Nj[w] & ~Ni[w];
                }
                Ui[j / 64] &= ~(1ULL << (j % 64));
                Uj[i / 64] &= ~(1ULL << (i % 64));
                int nzk = 0, nzw = 0, gamma_max = 0;
                for (int w = 0; w < W; w++) {
                    uint64_t m = Ui[w];
                    while (m) {
                        int k = w * 64 + __builtin_ctzll(m);
                        m &= m - 1;
                        int cnt = 0;
                        for (int u = 0; u < W; u++)
                            cnt += __builtin_popcountll(B[(size_t)k * W + u] & Uj[u]);
                        if (cnt) {
                            nzk++;
                            if (cnt > gamma_max)
                                gamma_max = cnt;
                        }
                    }
                }
                for (int w = 0; w < W; w++) {
                    uint64_t m = Uj[w];
                    while (m) {
                        int k = w * 64 + __builtin_ctzll(m);
                        m &= m - 1;
                        int cnt = 0;
                        for (int u = 0; u < W; u++)
                            cnt += __builtin_popcountll(B[(size_t)k * W + u] & Ui[u]);
                        if (cnt) {
                            nzw++;
                            if (cnt > gamma_max)
                                gamma_max = cnt;
                        }
                    }
                }
                double dmax = di > dj ? di : dj, dmin = di < dj ? di : dj;
                double S = gamma_max > 0 ? (nzk + nzw) / (gamma_max * dmax) : 0.0;
                double J = -0.5 * tri * fabs(1.0 / di - 1.0 / dj);
                double bf = 2.0 / di + 2.0 / dj - 2 + 2.0 * tri / dmax + tri / dmin + S;
                total += sign * bf;
                S_total += sign * S;
                J_total += sign * J;
            }
        }
        printf("%ld %.10f %.10f %.10f\n", s0 + smp, total, S_total, J_total);
        fflush(stdout);
        free(X);
        free(Y);
        free(B);
        free(head);
        free(next);
        free(EI);
        free(EJ);
        free(present);
        free(Ni);
        free(Nj);
        free(Ui);
        free(Uj);
    }
    return 0;
}
