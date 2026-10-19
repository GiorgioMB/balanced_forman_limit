/* E D^2_{x,y} F for BF curvature on the plane (Palm, r=1, intensity lam/pi).
   usage: d2c lam eps seed0 nsamp W NW   -> prints "seed tot S J" per sample (partial sums over edges e%NW==W) */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

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

static int poisson(double mu) { /* normal approx w/ correction fine for mu large; use exact for small */
    if (mu < 50) {
        double L = exp(-mu), p = 1;
        int k = 0;
        do {
            k++;
            p *= unif();
        } while (p > L);
        return k - 1;
    }

    double u1 = unif(), u2 = unif();
    double z = sqrt(-2 * log(u1 + 1e-300)) * cos(2 * M_PI * u2);
    int k = (int)floor(mu + sqrt(mu) * z + 0.5);
    return k < 0 ? 0 : k;
}

int main(int argc, char **argv) {
    double lam = atof(argv[1]), eps = atof(argv[2]);
    long s0 = atol(argv[3]);
    int n_samples = atoi(argv[4]), W = atoi(argv[5]), NW = atoi(argv[6]);

    const double R = 3.3 + eps;

    for (int smp = 0; smp < n_samples; smp++) {
        seed(1000003ULL * (s0 + smp) + 17);

        int N = poisson(lam * R * R);
        int NT = N + 2;
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
        int ix = N, iy = N + 1, WD = (NT + 63) / 64;

        uint64_t *B = calloc((size_t)NT * WD, 8);

        int G = (int)ceil(2 * R) + 1; /* grid cells of size 1 */
        int *head = malloc(sizeof(int) * G * G), *next_in_cell = malloc(sizeof(int) * NT);
        for (int i = 0; i < G * G; i++)
            head[i] = -1;
        for (int i = 0; i < NT; i++) {
            int cx = (int)floor(X[i] + R), cy = (int)floor(Y[i] + R);
            int c = cx * G + cy;
            next_in_cell[i] = head[c];
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

                    for (int j = head[ax * G + ay]; j >= 0; j = next_in_cell[j]) {
                        if (j <= i)
                            continue;
                        double dx = X[i] - X[j], dy = Y[i] - Y[j];
                        if (dx * dx + dy * dy <= 1.0) {
                            B[(size_t)i * WD + j / 64] |= 1ULL << (j % 64);
                            B[(size_t)j * WD + i / 64] |= 1ULL << (i % 64);

                            int i_near = (X[i] * X[i] + Y[i] * Y[i] <= (1 + eps) * (1 + eps)),
                                j_near = (X[j] * X[j] + Y[j] * Y[j] <= (1 + eps) * (1 + eps));
                            if ((i_near || j_near) && !(i >= N && j >= N)) {
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
        }

        uint64_t *pres = malloc(8 * WD), *Ni = malloc(8 * WD), *Nj = malloc(8 * WD),
                 *Ui = malloc(8 * WD), *Uj = malloc(8 * WD);
        double tot = 0, St = 0, Jt = 0;

        for (int c = 0; c < 4; c++) { /* 0:P+x+y 1:P+x 2:P+y 3:P */
            int hx = (c == 0 || c == 1), hy = (c == 0 || c == 2);
            double sg = (c == 0 || c == 3) ? 1 : -1;

            for (int w = 0; w < WD; w++)
                pres[w] = ~0ULL;
            if (!hx)
                pres[ix / 64] &= ~(1ULL << (ix % 64));
            if (!hy)
                pres[iy / 64] &= ~(1ULL << (iy % 64));

            for (int e = W; e < ne; e += NW) {
                int i = EI[e], j = EJ[e];
                if ((i == ix && !hx) || (j == ix && !hx) || (i == iy && !hy) || (j == iy && !hy))
                    continue;

                int di = 0, dj = 0, tri = 0;
                for (int w = 0; w < WD; w++) {
                    Ni[w] = B[(size_t)i * WD + w] & pres[w];
                    Nj[w] = B[(size_t)j * WD + w] & pres[w];
                    di += __builtin_popcountll(Ni[w]);
                    dj += __builtin_popcountll(Nj[w]);
                    tri += __builtin_popcountll(Ni[w] & Nj[w]);
                }
                if (di <= 1 || dj <= 1)
                    continue;

                for (int w = 0; w < WD; w++) {
                    Ui[w] = Ni[w] & ~Nj[w];
                    Uj[w] = Nj[w] & ~Ni[w];
                }
                Ui[j / 64] &= ~(1ULL << (j % 64));
                Uj[i / 64] &= ~(1ULL << (i % 64));

                int nzk = 0, nzw = 0, g = 0;
                for (int w = 0; w < WD; w++) {
                    uint64_t m = Ui[w];
                    while (m) {
                        int b = __builtin_ctzll(m);
                        m &= m - 1;
                        int k = w * 64 + b;
                        int cnt = 0;
                        for (int u = 0; u < WD; u++)
                            cnt += __builtin_popcountll(B[(size_t)k * WD + u] & Uj[u]);
                        if (cnt) {
                            nzk++;
                            if (cnt > g)
                                g = cnt;
                        }
                    }
                }
                for (int w = 0; w < WD; w++) {
                    uint64_t m = Uj[w];
                    while (m) {
                        int b = __builtin_ctzll(m);
                        m &= m - 1;
                        int k = w * 64 + b;
                        int cnt = 0;
                        for (int u = 0; u < WD; u++)
                            cnt += __builtin_popcountll(B[(size_t)k * WD + u] & Ui[u]);
                        if (cnt) {
                            nzw++;
                            if (cnt > g)
                                g = cnt;
                        }
                    }
                }

                double dmax = di > dj ? di : dj, dmin = di < dj ? di : dj;
                double S = g > 0 ? (nzk + nzw) / (g * dmax) : 0.0;
                double J = -0.5 * tri * fabs(1.0 / di - 1.0 / dj);
                double bf = 2.0 / di + 2.0 / dj - 2 + 2.0 * tri / dmax + tri / dmin + S;

                tot += sg * bf;
                St += sg * S;
                Jt += sg * J;
            }
        }

        printf("%ld %.10f %.10f %.10f\n", s0 + smp, tot, St, Jt);
        fflush(stdout);

        free(X);
        free(Y);
        free(B);
        free(head);
        free(next_in_cell);
        free(EI);
        free(EJ);
        free(pres);
        free(Ni);
        free(Nj);
        free(Ui);
        free(Uj);
    }
    return 0;
}
