#!/usr/bin/env python
import sys
raw = {
  'A': (2.0, 0.0 , -2.0),
  'C':(-2.0, 0.0 , -2.0),
  'G': (0.0, 2.0 , 2.0),
  'T': (0.0, -2.0 , 2.0)
}
states = 'ACGT'


def write_coord(k, d):
  s = '\coordinate [] ({}) at ({}*\\xfactor, {}*\\yfactor, {}*\\zfactor);\n'
  sys.stdout.write(s.format(k, d[0], d[1], d[2]))


# three state combos
three_state_keys = []
for f in states:
  for s in states:
    if states.index(f) >= states.index(s):
      continue
    for t in states:
      if states.index(s) >= states.index(t):
        continue
      for last in [f, s, t]:
        d = [0.0] * 3
        for i in range(3):
          d[i] = (raw[f][i] + raw[s][i] + raw[t][i] + raw[last][i])/4.0
        kl = [f, s, t, last]
        kl.sort()
        k = ''.join(kl)
        write_coord(k, d)
        three_state_keys.append(k)

print ','.join(three_state_keys)

for k in three_state_keys:
  kl = list(k)
  ks = set(kl)
  #sys.stdout.write('\\draw[opacity=.5] ({}) -- (ACGT);\n'.format(k))
  ns = set()
  for i in range(4):
    for el in ks:
      c = list(kl)
      c[i] = el
      c.sort()
      ns.add(''.join(c))
  ns = list(ns)
  ns.sort()
  for n in ns:
    if n != k:
      sys.stdout.write('\\draw[red, opacity=.5, very thin] ({}) -- ({});\n'.format(k,n))
  #print kl
