/* Shared helpers: the xoshiro256** generator seeded by splitmix64, an exact Poisson sampler, and the
   lens function f_2.  The generator and the seeding are those used for the paper's runs. */
#ifndef BFCOMMON_H
#define BFCOMMON_H

#include <math.h>
#include <stdint.h>

static uint64_t rng_state[4];

static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t next_u64(void) {
    const uint64_t result = rotl(rng_state[1] * 5, 7) * 9;
    const uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = rotl(rng_state[3], 45);
    return result;
}

/* uniform on [0,1) with 53 random bits */
static double unif(void) { return (next_u64() >> 11) * 0x1.0p-53; }

static void seed_rng(uint64_t z) {
    for (int i = 0; i < 4; i++) {
        z += 0x9e3779b97f4a7c15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        rng_state[i] = x ^ (x >> 31);
    }
}

/* Exact Poisson(mu) sampling: inversion by sequential search for mu <= 500, and the sum of
   independent Poisson variables with means <= 500 above (a sum of independent Poisson variables is
   Poisson).  O(mu) operations, which is negligible next to the curvature computation. */
static long poisson_small(double mu) {
    double u = unif(), p = exp(-mu), cdf = p;
    long k = 0;
    while (u > cdf && k < 100000) {
        k++;
        p *= mu / k;
        cdf += p;
    }
    return k;
}

static long poisson(double mu) {
    long k = 0;
    while (mu > 500.0) {
        k += poisson_small(500.0);
        mu -= 500.0;
    }
    return k + poisson_small(mu);
}

/* f_2(t): area of the intersection of two unit discs at distance t, divided by pi (Lemma 2.2) */
static inline double f2_lens(double t) {
    return (2.0 / M_PI) * (acos(t / 2) - (t / 2) * sqrt(1 - t * t / 4));
}

#endif
