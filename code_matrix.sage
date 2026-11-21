# Tribonacci numbers mod p and p^2 via the companion matrix
#
#   M = [[1,1,1],[1,0,0],[0,1,0]],  M^n = [[T_{n+1}, *, *], [T_n, *, *], [T_{n-1}, *, *]]
#
# with T_0 = 0, T_1 = 1, T_2 = 1.  N0(p) is the smallest n > 0 with
# T_{n-1} ≡ T_n ≡ 0 (mod p), i.e. the smallest n for which M^n is a scalar
# matrix mod p.  The flags p21, p22 record whether T_{N0-1}, resp. T_{N0},
# vanish mod p^2.
#
# Usage:  sage code_matrix.sage [LIMIT]     (default LIMIT = 10^7)

import os, sys, time

# Matrices are plain Python lists of ints: at p ~ 10^7, p^2 is too large for
# Sage's fast modular matrices, and pure-Python 3x3 arithmetic is ~5x faster.
def mat_mul(A, B, m):
  return [[(A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j]) % m
           for j in range(3)] for i in range(3)]

def mat_pow(n, m):
  """M^n mod m."""
  R = [[1,0,0],[0,1,0],[0,0,1]]
  A = [[1,1,1],[1,0,0],[0,1,0]]
  n = int(n); m = int(m)
  while n:
    if n & 1:
      R = mat_mul(R, A, m)
    A = mat_mul(A, A, m)
    n >>= 1
  return R

def trib(n, m):
  """T_n mod m."""
  return mat_pow(n, m)[1][0]

def is_scalar(A):
  return (A[0][1] == A[0][2] == A[1][0] == A[1][2] == A[2][0] == A[2][1] == 0
          and A[0][0] == A[1][1] == A[2][2])

def N_bound_factored(p):
  """Factorisation of a multiple of N0(p), from the number of roots of x^3-x^2-x-1 mod p."""
  _.<T> = GF(p)[]
  k = len((T^3 - T^2 - T - 1).roots(multiplicities=False))
  if k == 3:
    return factor(p - 1)
  if k == 1:
    return factor(p - 1) * factor(p + 1)   # p^2 - 1, factored in two halves
  if k == 0:
    return factor(p^2 + p + 1)
  raise ValueError(f"p = {p} divides the discriminant -44")

def N0(p):
  """(N0(p), p21, p22) for a prime p not dividing 44."""
  F = N_bound_factored(p)
  n = F.value()
  for q, e in F:                 # standard order-finding: strip each q while M^(n/q) is still scalar
    for _ in range(e):
      if is_scalar(mat_pow(n // q, p)):
        n //= q
      else:
        break
  B = mat_pow(n, p^2)
  return n, int(B[2][0] == 0), int(B[1][0] == 0)

@parallel(max(os.cpu_count() - 2, 1))   # leave 2 cores free
def scan_chunk(lo, hi):
  """(number of primes scanned, hits) for primes lo <= p < hi, excluding 2 and 11.

  Each worker generates its own primes, so the parent never holds the full
  list (hundreds of MB at 10^8, duplicated across forked workers)."""
  count, hits = 0, []
  for p in prime_range(lo, hi):
    if p in (2, 11):
      continue
    count += 1
    n, p21, p22 = N0(p)
    if p21 or p22:
      hits.append((p, n, p21, p22))
  return count, hits

def scan(limit, chunk_width=10^5):
  """All primes p < limit (excluding 2 and 11) with p21 = 1 or p22 = 1."""
  bounds = [(lo, min(lo + chunk_width, limit)) for lo in range(0, limit, chunk_width)]
  count, hits, done = 0, [], 0
  for _, (c, chunk_hits) in scan_chunk(bounds):
    count += c
    hits += chunk_hits
    done += 1
    if done % 25 == 0 or done == len(bounds):
      print(f"  {done}/{len(bounds)} chunks done", file=sys.stderr, flush=True)
  return count, sorted(hits)

# Run only for `sage code_matrix.sage`, not on load() (see code.sage for why
# __name__ can't be used).
if os.path.basename(sys.argv[0]) in ("code_matrix.sage", "code_matrix.sage.py"):
  limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10^7
  start = time.time()
  count, hits = scan(limit)
  elapsed = time.time() - start
  print(f"Primes p < {limit} (excluding 2, 11) with T_(N0-1) or T_(N0) ≡ 0 (mod p^2):")
  print("p, N0(p), p21, p22")
  for p, n, p21, p22 in hits:
    print(f"{p}, {n}, {p21}, {p22}")
  print(f"{len(hits)} found among {count} primes; "
        f"elapsed {elapsed:.1f} s on {max(os.cpu_count() - 2, 1)} worker processes")
