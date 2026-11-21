_.<x> = QQ[]
F.<b> = (x^3-x^2-x-1).splitting_field(simplify_all=True)
R = F.ring_of_integers()
l = (x^3-x^2-x-1).roots(ring=R, multiplicities=False)
dl = l[2]/l[0]-l[1]/l[0]-l[2]/l[1]+l[0]/l[1]+l[1]/l[2]-l[0]/l[2]
c = [(1/l[1]-1/l[2])/dl, (1/l[2]-1/l[0])/dl, (1/l[0]-1/l[1])/dl]
D = lcm([ci.denominator() for ci in c])  # = 22; D*c_i is integral

def power_mod(base, exp, mod):
  # NB: must be F.ideal(mod), not mod*R -- the latter is a generic order ideal
  # whose .reduce() returns its input unchanged (no reduction at all).
  I = F.ideal(mod)
  result = R(1)
  base = I.reduce(R(base))
  while exp > 0:
    if exp % 2 == 1:  # If exp is odd
      result = I.reduce(result * base)
    base = I.reduce(base * base)
    exp //= 2
  return result

def trib(n, m=0): # Tribonacci number T_n mod m, better for larger n
  if m == 0:
    return QQ(c[0]*l[0]^n + c[1]*l[1]^n + c[2]*l[2]^n)
  if gcd(m, D) != 1:
    # the c_i have denominator D = 22 (dl^2 = -44), so the Binet formula
    # can't be used mod m; compute T_n exactly and reduce instead.
    return trib(n) % m
  # The reduced powers r_i are only congruent to l_i^n mod mR, so
  # sum c_i r_i is not rational; but D*sum c_i r_i is integral and
  # congruent to the integer D*T_n mod mR, so it reduces to that integer.
  I = F.ideal(m)
  S = D * sum(c[i]*power_mod(l[i], n, m) for i in range(3))
  return ZZ(I.reduce(S)) * inverse_mod(D, m) % m

def N_bound(p): # N0 is a divisor of this number, which is the number of roots of the polynomial x^3-x^2-x-1 mod p
  _.<T> = GF(p)[]
  n_roots = len((T^3-T^2-T-1).roots(ring=GF(p), multiplicities=False))
  if n_roots == 3:
    return p-1
  elif n_roots == 1:
    return p^2-1
  elif n_roots == 0:
    return p^2+p+1
  else:
    raise ValueError(f"Unexpected number of roots for p = {p}: {n_roots}")

# Find the smallest n such that T_n ≡ 0 (mod p) and T_{n-1} ≡ 0 (mod p)
@parallel(max(os.cpu_count()-2,1))  # leave 2 cores free for other tasks
def N0(p):
  if not p.is_prime():
    raise ValueError(f"Input to N0 must be prime, got {p}")
  # let's try looking through divisors from the top down instead
  nb = N_bound(p)
  primes = nb.prime_divisors()
  # we know that it works for nb=N_bound(p), but we now repeatedly look at
  # the largest divisors of nb to possibly replace it by a smaller number
  found = True
  while found:
    found = False
    for q in primes:
      if power_mod(R(l[0]/l[1]), nb//q, p) == 1 and power_mod(R(l[1]/l[2]), nb//q, p) == 1:
      # if trib(nb//q, p) == 0 and trib(nb//q - 1, p) == 0:
        found = True
        if nb % q^2 != 0:
          primes.remove(q)  # no need to check this prime again
        nb //= q
        break
  print(f"N0({p}) = {nb}", end=", ")
  p21 = int(trib(nb-1, p^2) == 0)  # 1 or 0
  p22 = int(trib(nb, p^2) == 0)    # 1 or 0
  if p21 or p22:
    print(f"\np = {p}, n = {nb}, {p21}{p22}")
  return (nb,str(p21)+str(p22))


# Run the block only for `sage code.sage`, not on load(). __name__ can't be used:
# Sage >= 10.9 sets it to 'sage.all', and in Jupyter it is '__main__'. Check the
# script name instead -- older Sage runs a preparsed copy named code.sage.py.
import os, sys
if os.path.basename(sys.argv[0]) in ("code.sage", "code.sage.py"):
  # [(p,trib(N_bound(p), p)) for p in primes(start=13,stop=100)]
  print("Section 2.2 calculation:")
  zeroes = []
  for n in range(110):
    if trib(n,11) == 0:
      print(f"\tT_{n} ≡ 0 (mod 11),", end=" ")
      zeroes.append(n)
      if trib(n,11^2) != trib(n+110,11^2):
        print("can lift")
      else:
        print("cannot lift")
  
  print("Section 4 calculation:")
  print([(p[0][0],q) for p,q in N0([p for p in primes(start=13,stop=100_000_000)]) if q != '00'])