"""Self-test: every implementation of BF curvature against a literal implementation of Definition 2.1.

The reference implementation below works on Python sets and follows the definition word for word; in
particular it computes gamma_max from sq_j itself, so the tests also check identity (2.3), which the
fast implementations use. The C programs are tested on the same graphs by replaying their random number
generator (xoshiro256** seeded by splitmix64, and the exact Poisson sampler) in Python.
"""

import math

import numpy as np

from . import graphs, planted, theory, vertex
from .pipeline import bfgraph_binned, bfgraph_edges, kernel_samples


# ------------------------------------------------------------------------------- Definition 2.1
def bf_reference(adj, i, j):
    """Returns (BF, S, kink) of the edge {i, j}; adj maps a vertex to its set of neighbours."""
    N_i, N_j = adj[i], adj[j]
    d_i, d_j = len(N_i), len(N_j)
    if min(d_i, d_j) <= 1:
        return 0.0, 0.0, 0.0
    tri = len(N_i & N_j)
    sq_i = {k for k in N_i - N_j - {j} if adj[k] & (N_j - N_i - {i})}
    sq_j = {w for w in N_j - N_i - {i} if adj[w] & (N_i - N_j - {j})}
    gamma_max = max([len(adj[k] & sq_j) for k in sq_i] + [len(adj[w] & sq_i) for w in sq_j] + [0])
    four = (len(sq_i) + len(sq_j)) / (gamma_max * max(d_i, d_j)) if gamma_max else 0.0
    bf = 2 / d_i + 2 / d_j - 2 + 2 * tri / max(d_i, d_j) + tri / min(d_i, d_j) + four
    return bf, four, 0.5 * tri * abs(1 / d_i - 1 / d_j)


def adjacency_sets(n, close):
    adj = {v: set() for v in range(n)}
    for a in range(n):
        for b in range(a + 1, n):
            if close(a, b):
                adj[a].add(b)
                adj[b].add(a)
    return adj


# ------------------------------------------------------------------------------- the C generator
MASK = (1 << 64) - 1


class Xoshiro:
    def __init__(self, z):
        self.s = []
        for _ in range(4):
            z = (z + 0x9E3779B97F4A7C15) & MASK
            x = z
            x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
            self.s.append(x ^ (x >> 31))

    def unif(self):
        s = self.s
        result = ((((s[1] * 5) & MASK) << 7 | ((s[1] * 5) & MASK) >> 57) & MASK) * 9 & MASK
        t = (s[1] << 17) & MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = ((s[3] << 45) | (s[3] >> 19)) & MASK
        return (result >> 11) * 2.0**-53

    def poisson(self, mu):
        def small(m):
            u, p = self.unif(), math.exp(-m)
            cdf, k = p, 0
            while u > cdf and k < 100000:
                k += 1
                p *= m / k
                cdf += p
            return k
        k = 0
        while mu > 500.0:
            k += small(500.0)
            mu -= 500.0
        return k + small(mu)


# ------------------------------------------------------------------------------- the tests
def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_whole_graph(rng):
    n, r = 160, 0.12
    pts = rng.random((n, 2))
    pairs = graphs.surface_edges("T2", pts, r)
    fast = graphs.bf_edges(graphs.csr_from_pairs(pairs, n), pairs)
    adj = {v: set() for v in range(n)}
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    return all(_close(fast[e, 0], bf_reference(adj, a, b)[0]) and _close(fast[e, 1], bf_reference(adj, a, b)[1])
               for e, (a, b) in enumerate(pairs)), len(pairs)


def _planted_check(trial_fn, replay_fn, dist_fn, seeds):
    """Run a planted-edge sampler, replay its random draws, and compare with the reference."""
    checked = 0
    for seed in seeds:
        bf, four, kink = trial_fn(np.random.default_rng(seed))
        points, p, q, r = replay_fn(np.random.default_rng(seed))
        allpts = list(points) + [p, q]
        n = len(allpts)
        adj = adjacency_sets(n, lambda a, b: dist_fn(allpts[a], allpts[b]) <= r)
        ref = bf_reference(adj, n - 2, n - 1)
        if not (_close(bf, ref[0]) and _close(four, ref[1]) and _close(kink, ref[2])):
            return False, checked
        checked += 1
    return True, checked


def test_planted():
    results = {}
    lam, t, r = 9.0, 0.7, 0.5

    def replay_flat(rng, d=2):
        n = lam / theory.omega(d)
        lo = np.array([-t / 2 - 1] + [-1.0] * (d - 1))
        pts = rng.uniform(lo, -lo, size=(rng.poisson(n * np.prod(-2 * lo)), d))
        p = np.zeros(d)
        p[0] = -t / 2
        return pts, p, -p, 1.0

    results["flat"] = _planted_check(lambda g: planted.flat_trial(g, 2, lam, t), replay_flat,
                                     lambda a, b: np.linalg.norm(a - b), range(40))

    def replay_sphere(rng, k=2):
        n = lam / (theory.omega(k) * r**k)
        amax = r + t * r / 2
        count = rng.poisson(n * planted.sphere_cap_volume(k, amax))
        polar = planted._cap_polar_angles(rng, k, amax, count)
        dirs = planted._unit_vectors(rng, count, k)
        pts = np.column_stack([np.cos(polar), np.sin(polar)[:, None] * dirs])
        p = np.zeros(k + 1)
        p[0], p[1] = np.cos(t * r / 2), -np.sin(t * r / 2)
        q = p.copy()
        q[1] = -p[1]
        return pts, p, q, r

    geodesic = lambda a, b: math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))
    results["S2"] = _planted_check(lambda g: planted.sphere_trial(g, 2, lam, t, r), replay_sphere, geodesic, range(40))

    def replay_hyp(rng):
        n = lam / (np.pi * r**2)
        amax = r + t * r / 2
        count = rng.poisson(n * 2 * np.pi * (np.cosh(amax) - 1))
        rad = np.arccosh(1 + rng.random(count) * (np.cosh(amax) - 1))
        ang = 2 * np.pi * rng.random(count)
        pts = np.column_stack([np.cosh(rad), np.sinh(rad) * np.cos(ang), np.sinh(rad) * np.sin(ang)])
        p = np.array([np.cosh(t * r / 2), -np.sinh(t * r / 2), 0.0])
        q = p.copy()
        q[1] = -p[1]
        return pts, p, q, r

    hyperbolic = lambda a, b: math.acosh(max(1.0, a[0] * b[0] - a[1] * b[1] - a[2] * b[2]))
    results["H2"] = _planted_check(lambda g: planted.hyperbolic_trial(g, lam, t, r), replay_hyp, hyperbolic, range(40))

    def replay_s2xr(rng, direction):
        n = lam / (theory.omega(3) * r**3)
        e = t * r
        if direction == "R":
            amax, s_lo, s_hi = r, -r - e / 2, r + e / 2
            p, q = np.array([1.0, 0, 0, -e / 2]), np.array([1.0, 0, 0, e / 2])
        else:
            amax, s_lo, s_hi = r + e / 2, -r, r
            p = np.array([np.cos(e / 2), -np.sin(e / 2), 0.0, 0.0])
            q = np.array([np.cos(e / 2), np.sin(e / 2), 0.0, 0.0])
        count = rng.poisson(n * 2 * np.pi * (1 - np.cos(amax)) * (s_hi - s_lo))
        polar = np.arccos(1 - rng.random(count) * (1 - np.cos(amax)))
        azim = 2 * np.pi * rng.random(count)
        heights = rng.uniform(s_lo, s_hi, count)
        pts = np.column_stack([np.cos(polar), np.sin(polar) * np.cos(azim), np.sin(polar) * np.sin(azim), heights])
        return pts, p, q, r

    product = lambda a, b: math.hypot(math.acos(max(-1.0, min(1.0, float(a[:3] @ b[:3])))), a[3] - b[3])
    for direction in ("S", "R"):
        results[f"S2xR, along {direction}"] = _planted_check(lambda g: planted.s2xr_trial(g, lam, t, r, direction),
                                                             lambda g: replay_s2xr(g, direction), product, range(30))
    return results


def _tangent_distance(geometry):
    """Geodesic distance from tangent coordinates at x, independently of bflib.vertex."""
    def sphere_point(tan):
        rho = np.linalg.norm(tan)
        return np.concatenate([[np.cos(rho)], np.sin(rho) * tan / rho]) if rho > 0 else np.eye(len(tan) + 1)[0]

    if geometry.startswith("flat"):
        return lambda a, b: float(np.linalg.norm(a - b))
    if geometry in ("S2", "S3"):
        return lambda a, b: math.acos(max(-1.0, min(1.0, float(sphere_point(a) @ sphere_point(b)))))
    if geometry == "H2":
        def hyp(tan):
            rho = np.linalg.norm(tan)
            return np.array([np.cosh(rho), *(np.sinh(rho) * tan / rho)]) if rho > 0 else np.array([1.0, 0, 0])
        return lambda a, b: math.acosh(max(1.0, float(hyp(a)[0] * hyp(b)[0] - hyp(a)[1:] @ hyp(b)[1:])))
    if geometry == "S2xR":
        return lambda a, b: math.hypot(
            math.acos(max(-1.0, min(1.0, float(sphere_point(a[:2]) @ sphere_point(b[:2]))))), a[2] - b[2])
    raise ValueError(geometry)


def test_vertex():
    out = {}
    for geometry, r in [("flat2", 1.0), ("S2", 0.5), ("H2", 0.5), ("S2xR", 0.5), ("S3", 0.5)]:
        ok, checked = True, 0
        dist = _tangent_distance(geometry)
        for seed in range(6):
            local = vertex.sample_local(np.random.default_rng(seed), geometry, 7.0, r)
            A = vertex.adjacency(local)
            tan = local["tan"]
            adj = adjacency_sets(len(tan), lambda a, b: dist(tan[a], tan[b]) <= r)
            ok &= all(set(A.indices[A.indptr[v] : A.indptr[v + 1]]) == adj[v] for v in adj)
            js = sorted(adj[0])
            fast = vertex.bf_vertex_edges(A, 0, np.array(js, int))
            for e, j in enumerate(js):
                ref = bf_reference(adj, 0, j)
                ok &= _close(fast[e, 0], ref[0]) and _close(fast[e, 1], ref[1])
                checked += 1
        out[geometry] = (bool(ok), checked)
    return out


def _periodic(a):
    return a - math.floor(a + 0.5)


def test_c_edges():
    d, N, lam, seed = 2, 220, 9.0, 5
    gen = Xoshiro((1234567 * seed + 11) & MASK)
    X = [gen.unif() for _ in range(N * d)]
    r = math.pow(lam / (N * math.pi), 1.0 / d)
    pts = [X[i * d : (i + 1) * d] for i in range(N)]
    d2 = lambda a, b: sum(_periodic(pts[a][k] - pts[b][k]) ** 2 for k in range(d))
    adj = adjacency_sets(N, lambda a, b: d2(a, b) <= r * r)
    ref = sorted((float(np.float32(math.sqrt(d2(a, b)) / r)), *bf_reference(adj, a, b)[:2])
                 for a in adj for b in adj[a] if a < b)
    out = sorted(map(tuple, bfgraph_edges(d, N, lam, seed).astype(float)))
    ok = len(ref) == len(out) and all(abs(x[0] - y[0]) < 1e-7 and abs(x[1] - y[1]) < 1e-5 and abs(x[2] - y[2]) < 1e-5
                                      for x, y in zip(ref, out))
    return ok, len(ref)


def _binned_reference(space, n0, param, rmode, seed, n_bins):
    gen = Xoshiro((1000003 * seed + 7) & MASK)
    N = gen.poisson(n0)
    if space == "sphere2":
        pts = []
        for _ in range(N):
            z, phi = 2 * gen.unif() - 1, 2 * math.pi * gen.unif()
            rho = math.sqrt(max(0.0, 1 - z * z))
            pts.append((rho * math.cos(phi), rho * math.sin(phi), z))
        r, chord = param, 2 * math.sin(param / 2)
        d2 = lambda a, b: sum((pts[a][k] - pts[b][k]) ** 2 for k in range(3))
        close = lambda a, b: d2(a, b) <= chord * chord
        length = lambda a, b: 2 * math.asin(min(1.0, math.sqrt(d2(a, b)) / 2)) / r
    else:
        X = [gen.unif() for _ in range(2 * N)]
        pts = [X[2 * i : 2 * i + 2] for i in range(N)]
        r = math.sqrt(param / (math.pi * (N if rmode == "realised" else n0)))
        d2 = lambda a, b: sum(_periodic(pts[a][k] - pts[b][k]) ** 2 for k in range(2))
        close = lambda a, b: d2(a, b) <= r * r
        length = lambda a, b: math.sqrt(d2(a, b)) / r
    adj = adjacency_sets(N, close)
    bins = np.zeros((n_bins, 4))
    E = 0
    for a in adj:
        for b in adj[a]:
            if a < b:
                t = float(np.float32(length(a, b)))
                bf, _, kink = bf_reference(adj, a, b)
                k = min(int(t * n_bins), n_bins - 1)
                bins[k] += (1, bf, kink, -2 + 3 * (2 / math.pi) * (math.acos(t / 2) - (t / 2) * math.sqrt(1 - t * t / 4)))
                E += 1
    return N, E, bins


def test_c_binned():
    out = {}
    for space, n0, param, rmode in [("sphere2", 260.0, 0.35, "nominal"), ("torus2", 240.0, 8.0, "nominal"),
                                    ("torus2", 240.0, 8.0, "realised")]:
        row = bfgraph_binned(space, n0, param, rmode, 11, 1, 20)[0]
        N, E, bins = _binned_reference(space, n0, param, rmode, 11, 20)
        c = row[3:].reshape(20, 4)
        ok = int(row[1]) == N and int(row[2]) == E and np.array_equal(c[:, 0], bins[:, 0]) \
            and np.allclose(c[:, 1:], bins[:, 1:], rtol=1e-9, atol=1e-9)
        out[f"{space}/{rmode}"] = (bool(ok), E)
    return out


def test_c_kernel():
    lam, eps, seed = 7.0, 0.3, 3
    gen = Xoshiro((1000003 * seed + 17) & MASK)
    R = 3.3 + eps
    N = gen.poisson(lam * R * R)
    pts = []
    for _ in range(N):
        rr, th = R * math.sqrt(gen.unif()), 2 * math.pi * gen.unif()
        pts.append((rr * math.cos(th), rr * math.sin(th)))
    pts += [(0.0, 0.0), (eps, 0.0)]
    ix, iy = N, N + 1
    close = lambda a, b: (pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2 <= 1.0
    near = lambda a: pts[a][0] ** 2 + pts[a][1] ** 2 <= (1 + eps) ** 2
    full = adjacency_sets(N + 2, close)
    edges = [(a, b) for a in full for b in full[a] if a < b and (near(a) or near(b)) and not (a >= N and b >= N)]
    totals = np.zeros(3)
    for has_x, has_y, sign in [(1, 1, 1), (1, 0, -1), (0, 1, -1), (0, 0, 1)]:
        absent = ({ix} if not has_x else set()) | ({iy} if not has_y else set())
        adj = {v: (nb - absent) for v, nb in full.items() if v not in absent}
        for a, b in edges:
            if a in absent or b in absent:
                continue
            bf, four, kink = bf_reference(adj, a, b)
            totals += sign * np.array([bf, four, -kink])
    got = kernel_samples(lam, eps, seed, 1)[0, 1:]
    return bool(np.allclose(got, totals, atol=1e-8)), len(edges)


def test_theory():
    checks = {}
    for d in (2, 3, 5):
        checks[f"E f_{d}(T) = 3 f_{d}(1)/2"] = _close(theory.mean_over_T(d, lambda t: float(theory.f_lens(d, t))),
                                                      1.5 * float(theory.f_lens(d, 1.0)), 1e-8)
    checks["vartheta_2 = 1/24"] = _close(theory.vartheta_d(2), 1 / 24, 1e-9)
    checks["vartheta_3 = 67/2560"] = _close(theory.vartheta_d(3), 67 / 2560, 1e-9)
    checks["eta_2 = 3 sqrt3/(32 pi) - 1/24"] = _close(theory.eta_d(2), 3 * math.sqrt(3) / (32 * math.pi) - 1 / 24, 1e-8)
    checks["eta_3 = 7/1280"] = _close(theory.eta_d(3), 7 / 1280, 1e-8)
    f2 = lambda t: (2 / math.pi) * (math.acos(t / 2) - t / 2 * math.sqrt(1 - t * t / 4))
    checks["f_2 closed form"] = all(_close(float(theory.f_lens(2, t)), f2(t), 1e-12) for t in (0.1, 0.5, 1.0, 1.7))
    checks["f_3 closed form"] = all(_close(float(theory.f_lens(3, t)), 1 - 0.75 * t + t**3 / 16, 1e-12)
                                    for t in (0.1, 0.5, 1.0, 1.7))
    rng = np.random.default_rng(0)
    f = 0.53
    z = rng.standard_normal((3, 2_000_000))
    w = 3 * math.sqrt(f) * (1 - f) * z[0] - 1.5 * f * math.sqrt(1 - f) * (z[1] + z[2]) \
        - 0.5 * f * math.sqrt(1 - f) * np.abs(z[1] - z[2])
    checks["moments of W_t (Theorem 3.2)"] = abs(w.mean() - theory.W_mean(f)) < 3e-3 and abs(w.var() / theory.W_var(f) - 1) < 3e-3
    return checks


def run():
    rng = np.random.default_rng(2026)
    failures = 0

    def report(name, ok, detail=""):
        nonlocal failures
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{(' (' + detail + ')') if detail else ''}", flush=True)

    print("Definition 2.1 against the implementations:")
    ok, n = test_whole_graph(rng)
    report("whole graphs, bflib.graphs.bf_edges", ok, f"{n} edges")
    for geometry, (ok, n) in test_planted().items():
        report(f"planted edges, {geometry}", ok, f"{n} edges")
    for geometry, (ok, n) in test_vertex().items():
        report(f"planted vertex, {geometry}", ok, f"{n} edges")
    ok, n = test_c_edges()
    report("C program bfgraph, per-edge mode", ok, f"{n} edges")
    for name, (ok, n) in test_c_binned().items():
        report(f"C program bfgraph, binned mode, {name}", ok, f"{n} edges")
    ok, n = test_c_kernel()
    report("C program kernel, second difference", ok, f"{n} edges")
    print("Closed forms:")
    for name, ok in test_theory().items():
        report(name, ok)
    print("all tests passed" if failures == 0 else f"{failures} test(s) FAILED")
    return failures
