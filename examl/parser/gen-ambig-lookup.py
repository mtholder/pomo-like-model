#!/usr/bin/env python
from __future__ import print_function
states = 'ACGT'
# Order of states in PoMo A, C, G, T, AC, AG, AT, CG, CT, GT
# converted to binary codes
model_state_bin = [1, 2, 4, 8, 3, 5, 9, 6, 10, 12]
state2bit = dict((s, 1<<n) for n, s in enumerate(states))
int2state = dict((v, k) for k, v in state2bit.items())
ambig = {}
for b in range(1, 16):
  sl = [s for i, s in int2state.items() if 0 != (i & b)]
  sl.sort()
  ambig[b] = ''.join(sl)
int2state.update(ambig)

data2effect = {}
for b, s in int2state.items():
  effects = []
  for model_bit in model_state_bin:
    model_state = int2state[model_bit]
    if len(model_state) > 2:
      effects.append(1)
    elif len(model_state) == 1:
      if model_state in s:
        effects.append(4)
      else:
        effects.append(1)
    else:
      assert(len(model_state) == 2)
      if model_state[0] in s:
        if model_state[1] in s:
          effects.append(0)
        else:
          effects.append(4)
      elif model_state[1] in s:
         effects.append(2)
      else:
         effects.append(1)
  data2effect[b] = effects


k = int2state.keys()
k.sort()
sorted_codes = [int2state[i] for i in k]
#print('           ' + '   '.join([i.ljust(4) for i in sorted_codes]))
print('//   ' + ', '.join([int2state[i].rjust(2) for i in model_state_bin]))
print('{%s},' % ', '.join(['15']*11))
for b, e in data2effect.items():
  print('{ 0, %s},' % ', '.join([str(i).rjust(2) for i in e]))
  #print(str(int2state[b]).rjust(4), ' ->', e)