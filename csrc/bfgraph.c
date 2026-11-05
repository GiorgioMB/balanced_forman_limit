/* Exact Balanced Forman curvature of every edge of a random geometric graph.

   bfgraph edges d N lam seed outfile
       Binomial RGG: N uniform points on the unit torus [0,1)^d, d in {2,3}, radius
       r = (lam / (N omega_d))^(1/d).  Writes float32 triples (t, BF, S) for every edge, where t is the
       normalised length and S the 4-cycle term.  (Table S2, Figure S2.)

   bfgraph binned space n0 param rmode seed0 ngraphs nbins
       Poisson RGGs with Poisson(n0) points, graphs with seeds seed0, ..., seed0 + ngraphs - 1.
       space = torus2: unit torus [0,1)^2, param = lam, and r = sqrt(lam / (pi n0)) (rmode nominal) or
                       r = sqrt(lam / (pi N)) for the realised number N of points (rmode realised).
       space = sphere2: unit sphere S^2 with geodesic radius r = param (rmode ignored).
       For each graph prints one line: seed N E, then for each of nbins bins of t in (0,1]:
       count, sum of BF, sum of kink and sum of Psi_2(t), where kink = |tri| |1/d_i - 1/d_j| / 2 is the
       max/min part, so that BF + kink is the symmetrised curvature.  (Tables 4, S5; Section 8.4.)

   The 4-cycle counts use identity (2.3): for k in U_i, |N(k) n sq_j| = |N(k) n U_j|. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bfcommon.h"

typedef struct {
    int n, d, words;
    uint64_t *adj; /* n x words bitset */
    long n_edges, cap;
    int *ei, *ej;
    float *et;
} Graph;

static void add_edge(Graph *g, int a, int b, double t) {
    uint64_t *row_a = g->adj + (size_t)a * g->words;
    if (row_a[b / 64] & (1ULL << (b % 64)))
        return;
    row_a[b / 64] |= 1ULL << (b % 64);
    g->adj[(size_t)b * g->words + a / 64] |= 1ULL << (a % 64);
    if (g->n_edges == g->cap) {
        g->cap *= 2;
        g->ei = realloc(g->ei, sizeof(int) * g->cap);
        g->ej = realloc(g->ej, sizeof(int) * g->cap);
        g->et = realloc(g->et, sizeof(float) * g->cap);
    }
    g->ei[g->n_edges] = a;
    g->ej[g->n_edges] = b;
    g->et[g->n_edges] = (float)t;
    g->n_edges++;
}

static void graph_init(Graph *g, int n, int d) {
    g->n = n;
    g->d = d;
    g->words = (n + 63) / 64;
    g->adj = calloc((size_t)n * g->words, 8);
    g->cap = 1 << 20;
    g->n_edges = 0;
    g->ei = malloc(sizeof(int) * g->cap);
    g->ej = malloc(sizeof(int) * g->cap);
    g->et = malloc(sizeof(float) * g->cap);
}

static void graph_free(Graph *g) {
    free(g->adj);
    free(g->ei);
    free(g->ej);
    free(g->et);
}

/* Sort points by cell so that the neighbourhoods occupy few bitset words.  cell[] gives the cell of
   each point; returns the permutation and fills cell_start (size n_cells + 1). */
static int *sort_by_cell(const int *cell, int n, int n_cells, int *cell_start) {
    memset(cell_start, 0, sizeof(int) * (n_cells + 1));
    for (int i = 0; i < n; i++)
        cell_start[cell[i] + 1]++;
    for (int c = 0; c < n_cells; c++)
        cell_start[c + 1] += cell_start[c];
    int *fill = malloc(sizeof(int) * (n_cells + 1));
    memcpy(fill, cell_start, sizeof(int) * (n_cells + 1));
    int *order = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++)
        order[fill[cell[i]]++] = i;
    free(fill);
    return order;
}

/* BF curvature of the edge {i,j}; returns 0 for all three outputs if min degree <= 1. */
typedef struct {
    uint64_t *Ui, *Uj;
    int *nz_i, *nz_j;
} Scratch;

static void edge_bf(const Graph *g, Scratch *s, int i, int j, double *bf, double *sq, double *kink) {
    const int W = g->words;
    const uint64_t *Ni = g->adj + (size_t)i * W, *Nj = g->adj + (size_t)j * W;
    int di = 0, dj = 0, tri = 0;
    for (int w = 0; w < W; w++) {
        di += __builtin_popcountll(Ni[w]);
        dj += __builtin_popcountll(Nj[w]);
        tri += __builtin_popcountll(Ni[w] & Nj[w]);
        s->Ui[w] = Ni[w] & ~Nj[w];
        s->Uj[w] = Nj[w] & ~Ni[w];
    }
    *bf = *sq = *kink = 0.0;
    if (di <= 1 || dj <= 1)
        return;
    s->Ui[j / 64] &= ~(1ULL << (j % 64));
    s->Uj[i / 64] &= ~(1ULL << (i % 64));
    int nnz_i = 0, nnz_j = 0;
    for (int w = 0; w < W; w++) {
        if (s->Ui[w])
            s->nz_i[nnz_i++] = w;
        if (s->Uj[w])
            s->nz_j[nnz_j++] = w;
    }
    int nzk = 0, nzw = 0, gamma_max = 0;
    for (int a = 0; a < nnz_i; a++) {
        uint64_t m = s->Ui[s->nz_i[a]];
        while (m) {
            int k = s->nz_i[a] * 64 + __builtin_ctzll(m);
            m &= m - 1;
            const uint64_t *Nk = g->adj + (size_t)k * W;
            int c = 0;
            for (int b = 0; b < nnz_j; b++)
                c += __builtin_popcountll(Nk[s->nz_j[b]] & s->Uj[s->nz_j[b]]);
            if (c) {
                nzk++;
                if (c > gamma_max)
                    gamma_max = c;
            }
        }
    }
    for (int a = 0; a < nnz_j; a++) {
        uint64_t m = s->Uj[s->nz_j[a]];
        while (m) {
            int k = s->nz_j[a] * 64 + __builtin_ctzll(m);
            m &= m - 1;
            const uint64_t *Nk = g->adj + (size_t)k * W;
            int c = 0;
            for (int b = 0; b < nnz_i; b++)
                c += __builtin_popcountll(Nk[s->nz_i[b]] & s->Ui[s->nz_i[b]]);
            if (c) {
                nzw++;
                if (c > gamma_max)
                    gamma_max = c;
            }
        }
    }
    double dmax = di > dj ? di : dj, dmin = di < dj ? di : dj;
    *sq = gamma_max > 0 ? (nzk + nzw) / (gamma_max * dmax) : 0.0;
    *bf = 2.0 / di + 2.0 / dj - 2 + 2.0 * tri / dmax + tri / dmin + *sq;
    *kink = 0.5 * tri * fabs(1.0 / di - 1.0 / dj);
}

static inline double periodic(double a) { return a - floor(a + 0.5); }

/* RGG on the unit torus [0,1)^d (d = 2, 3) for points X (n x d, row major). */
static void build_torus(Graph *g, const double *X, int n, int d, double r) {
    int G = (int)floor(1.0 / r);
    if (G < 3) {
        fprintf(stderr, "bfgraph: radius too large for the cell list\n");
        exit(1);
    }
    double side = 1.0 / G;
    int n_cells = 1;
    for (int k = 0; k < d; k++)
        n_cells *= G;
    int *cell = calloc(n > 0 ? n : 1, sizeof(int)), *start = malloc(sizeof(int) * (n_cells + 1));
    for (int i = 0; i < n; i++) {
        int c = 0;
        for (int k = 0; k < d; k++) {
            int ck = (int)(X[i * d + k] / side);
            if (ck >= G)
                ck = G - 1;
            c = c * G + ck;
        }
        cell[i] = c;
    }
    int *order = sort_by_cell(cell, n, n_cells, start);
    double *Y = malloc(sizeof(double) * n * d);
    for (int a = 0; a < n; a++)
        for (int k = 0; k < d; k++)
            Y[a * d + k] = X[order[a] * d + k];
    graph_init(g, n, d);
    int n_neighbours = d == 2 ? 9 : 27;
    for (int c = 0; c < n_cells; c++) {
        int coord[3], rest = c;
        for (int k = d - 1; k >= 0; k--) {
            coord[k] = rest % G;
            rest /= G;
        }
        for (int m = 0; m < n_neighbours; m++) {
            int digits = m, c2 = 0;
            for (int k = 0; k < d; k++) {
                int off = digits % 3 - 1;
                digits /= 3;
                c2 = c2 * G + ((coord[k] + off + G) % G);
            }
            for (int a = start[c]; a < start[c + 1]; a++)
                for (int b = start[c2]; b < start[c2 + 1]; b++) {
                    if (b <= a)
                        continue;
                    double d2 = 0;
                    for (int k = 0; k < d; k++) {
                        double z = periodic(Y[a * d + k] - Y[b * d + k]);
                        d2 += z * z;
                    }
                    if (d2 <= r * r)
                        add_edge(g, a, b, sqrt(d2) / r);
                }
        }
    }
    free(cell);
    free(start);
    free(order);
    free(Y);
}

/* Geodesic RGG on the unit sphere S^2: chordal distance <= 2 sin(r/2). */
static void build_sphere(Graph *g, const double *X, int n, double r) {
    double chord = 2 * sin(r / 2);
    int G = (int)floor(2.0 / chord);
    if (G < 1)
        G = 1;
    double side = 2.0 / G;
    int n_cells = G * G * G;
    int *cell = calloc(n > 0 ? n : 1, sizeof(int)), *start = malloc(sizeof(int) * (n_cells + 1));
    for (int i = 0; i < n; i++) {
        int c = 0;
        for (int k = 0; k < 3; k++) {
            int ck = (int)((X[i * 3 + k] + 1.0) / side);
            if (ck >= G)
                ck = G - 1;
            if (ck < 0)
                ck = 0;
            c = c * G + ck;
        }
        cell[i] = c;
    }
    int *order = sort_by_cell(cell, n, n_cells, start);
    double *Y = malloc(sizeof(double) * n * 3);
    for (int a = 0; a < n; a++)
        for (int k = 0; k < 3; k++)
            Y[a * 3 + k] = X[order[a] * 3 + k];
    graph_init(g, n, 3);
    for (int c = 0; c < n_cells; c++) {
        int cx = c / (G * G), cy = (c / G) % G, cz = c % G;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) {
                    int x2 = cx + dx, y2 = cy + dy, z2 = cz + dz;
                    if (x2 < 0 || y2 < 0 || z2 < 0 || x2 >= G || y2 >= G || z2 >= G)
                        continue;
                    int c2 = (x2 * G + y2) * G + z2;
                    for (int a = start[c]; a < start[c + 1]; a++)
                        for (int b = start[c2]; b < start[c2 + 1]; b++) {
                            if (b <= a)
                                continue;
                            double d2 = 0;
                            for (int k = 0; k < 3; k++) {
                                double z = Y[a * 3 + k] - Y[b * 3 + k];
                                d2 += z * z;
                            }
                            if (d2 <= chord * chord)
                                add_edge(g, a, b, 2 * asin(fmin(1.0, sqrt(d2) / 2)) / r);
                        }
                }
    }
    free(cell);
    free(start);
    free(order);
    free(Y);
}

static Scratch scratch_new(int words) {
    Scratch s;
    s.Ui = malloc(8 * words);
    s.Uj = malloc(8 * words);
    s.nz_i = malloc(sizeof(int) * words);
    s.nz_j = malloc(sizeof(int) * words);
    return s;
}

static void scratch_free(Scratch *s) {
    free(s->Ui);
    free(s->Uj);
    free(s->nz_i);
    free(s->nz_j);
}

static int mode_edges(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: bfgraph edges d N lam seed outfile\n");
        return 2;
    }
    int d = atoi(argv[2]), N = atoi(argv[3]);
    double lam = atof(argv[4]);
    long sd = atol(argv[5]);
    double omega_d = (d == 2) ? M_PI : 4.0 * M_PI / 3.0;
    double r = pow(lam / (N * omega_d), 1.0 / d);
    seed_rng(1234567ULL * sd + 11);
    double *X = malloc(sizeof(double) * N * d);
    for (int i = 0; i < N * d; i++)
        X[i] = unif();
    Graph g;
    build_torus(&g, X, N, d, r);
    Scratch s = scratch_new(g.words);
    FILE *out = fopen(argv[6], "wb");
    if (!out) {
        fprintf(stderr, "bfgraph: cannot open %s\n", argv[6]);
        return 1;
    }
    for (long e = 0; e < g.n_edges; e++) {
        double bf, sq, kink;
        edge_bf(&g, &s, g.ei[e], g.ej[e], &bf, &sq, &kink);
        float rec[3] = {g.et[e], (float)bf, (float)sq};
        fwrite(rec, sizeof(float), 3, out);
    }
    fclose(out);
    fprintf(stderr, "edges %ld r %.5f\n", g.n_edges, r);
    scratch_free(&s);
    graph_free(&g);
    free(X);
    return 0;
}

static int mode_binned(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: bfgraph binned torus2|sphere2 n0 param nominal|realised seed0 ngraphs nbins\n");
        return 2;
    }
    const char *space = argv[2];
    double n0 = atof(argv[3]), param = atof(argv[4]);
    int realised = strcmp(argv[5], "realised") == 0;
    long seed0 = atol(argv[6]);
    int n_graphs = atoi(argv[7]), NB = atoi(argv[8]);
    int sphere = strcmp(space, "sphere2") == 0;
    if (!sphere && strcmp(space, "torus2") != 0) {
        fprintf(stderr, "bfgraph: unknown space %s\n", space);
        return 2;
    }
    long *count = malloc(sizeof(long) * NB);
    double *sum_bf = malloc(sizeof(double) * NB), *sum_kink = malloc(sizeof(double) * NB),
           *sum_psi = malloc(sizeof(double) * NB);
    for (int gi = 0; gi < n_graphs; gi++) {
        long seed = seed0 + gi;
        seed_rng(1000003ULL * (uint64_t)seed + 7);
        int N = (int)poisson(n0);
        double *X;
        Graph g;
        if (sphere) {
            X = malloc(sizeof(double) * N * 3);
            for (int i = 0; i < N; i++) {
                double z = 2 * unif() - 1, phi = 2 * M_PI * unif(), rho = sqrt(fmax(0.0, 1 - z * z));
                X[3 * i] = rho * cos(phi);
                X[3 * i + 1] = rho * sin(phi);
                X[3 * i + 2] = z;
            }
            build_sphere(&g, X, N, param);
        } else {
            X = malloc(sizeof(double) * N * 2);
            for (int i = 0; i < 2 * N; i++)
                X[i] = unif();
            double r = sqrt(param / (M_PI * (realised ? (double)N : n0)));
            build_torus(&g, X, N, 2, r);
        }
        memset(count, 0, sizeof(long) * NB);
        memset(sum_bf, 0, sizeof(double) * NB);
        memset(sum_kink, 0, sizeof(double) * NB);
        memset(sum_psi, 0, sizeof(double) * NB);
        Scratch s = scratch_new(g.words);
        for (long e = 0; e < g.n_edges; e++) {
            double bf, sq, kink, t = g.et[e];
            edge_bf(&g, &s, g.ei[e], g.ej[e], &bf, &sq, &kink);
            int b = (int)(t * NB);
            if (b >= NB)
                b = NB - 1;
            count[b]++;
            sum_bf[b] += bf;
            sum_kink[b] += kink;
            sum_psi[b] += -2 + 3 * f2_lens(t);
        }
        printf("%ld %d %ld", seed, N, g.n_edges);
        for (int b = 0; b < NB; b++)
            printf(" %ld %.12g %.12g %.12g", count[b], sum_bf[b], sum_kink[b], sum_psi[b]);
        printf("\n");
        fflush(stdout);
        scratch_free(&s);
        graph_free(&g);
        free(X);
    }
    free(count);
    free(sum_bf);
    free(sum_kink);
    free(sum_psi);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "edges") == 0)
        return mode_edges(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "binned") == 0)
        return mode_binned(argc, argv);
    fprintf(stderr, "usage: bfgraph edges ... | bfgraph binned ...\n");
    return 2;
}
