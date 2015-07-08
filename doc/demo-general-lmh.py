import numpy as np
from scipy import linalg
import itertools
import sys
TOL = 1e-8
NUM_STATES = 43
class NUC:
  A, C, G, T = range(4)
class GTR_RATE:
  AC, AG, AT, CG, CT, GT = range(6)
N = 12

_decode_1 = {'H': N,
             ' ': 0}
_decode_2 = {'H': N - 1,
             'M': N/2,
             'L': 1, 
             ' ': 0
            }
_decode_3 = {'H': N - 2,
             'M': N/3,
             'L': 1, 
             ' ': 0
            }
_decode_4 = {'H': N - 3,
             'M': N/4,
             'L': 1, 
             ' ': 0
            }
_decoders = {1: _decode_1,
             2: _decode_2,
             3: _decode_3,
             4: _decode_4,
            }

def debug(m):
  sys.stderr.write(m + '\n')
class GLMHPomoState(object):
  def __init__(self, code=None, A=None, C=None, G=None, T=None):
    self.index = None
    if code is not None:
      assert len(code) == 4
      acgt = []
      num_alleles = len(''.join(code.split(' ')))
      assert num_alleles > 0
      assert num_alleles <= 4
      _dec = _decoders[num_alleles]
      for letter in code:
        acgt.append(_dec[letter])
    else:
      acgt = [A, C, G, T]
    assert sum(acgt) == N
    self._n_vec = tuple(acgt)
    self.alleles = set([n for n, i in enumerate(acgt) if i > 0])
    self.alleles_list = [i for i in self.alleles]
    self._num_alleles = sum([1 for i in self._n_vec if i != 0])
    self._is_mid_freq = False
    if self._num_alleles > 1 and (N/self._num_alleles) in self._n_vec:
      self._is_mid_freq = True
    count_nuc_pairs = []
    self.allele2count = self._n_vec
    for n, c in enumerate(self._n_vec):
      count_nuc_pairs.append((c, n))
    count_nuc_pairs.sort()
    self.most_common_allele_idx = count_nuc_pairs[-1][1]
  @property
  def as_n_vec(self):
    return self._n_vec
  @property
  def num_alleles(self):
    return self._num_alleles
  def __str__(self):
    return 'GLMHPomoState(A={}, C={}, G={}, T={})'.format(*self.as_n_vec)
  def new_allele_idx(self, from_state):
    d = list(self.alleles - from_state.alleles)
    assert len(d) == 1
    return d[0]
  def is_adjacent(self, other):
    diff_num_alleles = other.num_alleles - self.num_alleles
    if (diff_num_alleles == 1) or  (diff_num_alleles == -1):
      if diff_num_alleles > 0:
        large, smaller = other.alleles, self.alleles
      else:
        large, smaller = self.alleles, other.alleles
      d = large - smaller
      if len(d) != 1:
        return False
      if self._is_mid_freq or other._is_mid_freq:
        return False
      return self.most_common_allele_idx == other.most_common_allele_idx
    elif diff_num_alleles == 0:
      return self.alleles == other.alleles
    return False
class GLMHPomoStateSpace(object):
  def __init__(self):
    codes = 'HML '
    all_states = []
    state_codes = set()
    for i in itertools.combinations_with_replacement(codes, 4):
      for j in itertools.permutations(i):
        code = ''.join(j)
        if code in state_codes:
          continue
        try:
          x = GLMHPomoState(code)
        except Exception as blah:
          pass
        else:
          state_codes.add(code)
          all_states.append(x)
    #print len(state_codes)
    n_s = [(4 - i.num_alleles, i.as_n_vec, i) for i in all_states]
    n_s.sort(reverse=True)
    self.all_states = [i[2] for i in n_s]
    for n, s in enumerate(all_states):
      s.index = n
      #print n, s
    self.num_states = len(self.all_states)
  def __getitem__(self, index):
    return self.all_states[index]


def sum_of_rate_pair_products(R, i, j, k):
  return R[i][j]*R[i][k] + R[i][j]*R[j][k] + R[i][k]*R[j][k]
class GLMHPomoModel(object):
  def __init__(self, **params):
    self.S = GLMHPomoStateSpace()
    self.nuc_freqs = params['nuc_freqs']
    F = self.nuc_freqs
    self.nuc_freq_product = F[0]*F[1]*F[2]*F[3]
    self.gtr_rates = params['gtr_rates']
    g = self.gtr_rates
    self.rev_mat = [[None, g[0], g[1], g[2], ],
                    [g[0], None, g[3], g[4], ],
                    [g[1], g[3], None, g[5], ],
                    [g[2], g[4], g[5], None, ],
                   ]
    R = self.rev_mat
    self.prob_diallelic = params['prob_diallelic']
    self.psi = params['psi']
    f2 = self.psi
    f3 = f2*f2
    f4= f2*f2*f2
    f1 = 1.0 - f2 -f3 - f4
    # K_2
    self.K2 = 0.0
    for i in xrange(4):
      for j in xrange(i + 1, 4):
        self.K2 += F[i]*F[j]*R[i][j]
    # K_3
    self.K3 = 0.0
    for i in xrange(4):
      for j in xrange(i + 1, 4):
        for k in xrange(j + 1, 4):
          fp =  F[i]*F[j]*F[k]
          rs = sum_of_rate_pair_products(R, i, j, k)
          self.K3 += fp*rs
    self.f1, self.f2, self.f3, self.f4 = f1, f2, f3, f4
    self.drift_rate = params['drift_rate']
    debug('K2 = {}'.format(self.K2))
    debug('K3 = {}'.format(self.K3))
    debug('f1 = {}'.format(self.f1))
    debug('f2 = {}'.format(self.f2))
    debug('f3 = {}'.format(self.f3))
    debug('f4 = {}'.format(self.f4))
  def calc_instantaneous_rate(self, from_state, to_state):
    if not from_state.is_adjacent(to_state):
      return 0.0
    diff_num_alleles = to_state.num_alleles - from_state.num_alleles
    if diff_num_alleles == 1:
      new_nuc_idx = to_state.new_allele_idx(from_state)
      from_nuc_idx = from_state.most_common_allele_idx
      assert new_nuc_idx != from_nuc_idx
      rev_mat_rate = self.rev_mat[new_nuc_idx][from_nuc_idx]
      dest_freq = self.nuc_freqs[new_nuc_idx]
      cardinality_factor = float(N + 1 - from_state.num_alleles)/N
      return rev_mat_rate*dest_freq*cardinality_factor
    elif diff_num_alleles == -1:
      lost_nuc_ind = from_state.new_allele_idx(to_state)
      if to_state.num_alleles == 1:
        return self.f1*self.K2*(2 + self.psi)/self.f2
      if to_state.num_alleles == 2:
        i, j = to_state.alleles_list
        if i != to_state.most_common_allele_idx:
          assert j == to_state.most_common_allele_idx
          i, j = j, i
        k = lost_nuc_ind
        nrf = self.rev_mat[i][j] * self.rev_mat[i][k]
        rs = sum_of_rate_pair_products(self.rev_mat, i, j, k)
        numerator = (N - 1)*self.K3*(3 + self.psi)*nrf
        denominator = N*self.K2*(2 + self.psi)*self.f2*rs
        return numerator/denominator
      assert to_state.num_alleles == 3
      i, j, k = to_state.alleles_list
      rs = sum_of_rate_pair_products(self.rev_mat, i, j, k)
      single_rate_fac = self.rev_mat[to_state.most_common_allele_idx][lost_nuc_ind]
      numerator = rs*single_rate_fac*self.nuc_freq_product*(4 + self.psi)*(N - 2)
      denominator = N * self.f2 * self.K3 * (3 + self.psi)
      return numerator/denominator
    else:
      if to_state.num_alleles == 2:
        i, j = to_state.alleles_list
        rf = self.rev_mat[i][j]
        fp = self.nuc_freqs[i]*self.nuc_freqs[j]
        numerator = self.f2*rf*fp
        denominator = self.K2*(2 + self.psi)
      elif to_state.num_alleles == 3:
        i, j, k = to_state.alleles_list
        rf = sum_of_rate_pair_products(self.rev_mat, i, j, k)
        fp = self.nuc_freqs[i]*self.nuc_freqs[j]*self.nuc_freqs[k]
        numerator = self.f3 * fp * rf
        denominator = self.K3*(3 + self.psi)
      else:
        assert to_state.num_alleles == 4
        numerator = self.f4
        denominator = (4 + self.psi)
      c = numerator/denominator
      c *= self.drift_rate
      if to_state._is_mid_freq:
        return c * self.psi
      return c
  def q_mat(self):
    S = self.S
    q = [[0]*S.num_states for i in xrange(S.num_states)]
    for from_ind, row in enumerate(q):
      from_state = S[from_ind]
      row_sum = 0.0
      for to_ind in xrange(S.num_states):
        if to_ind == from_ind:
          continue
        to_state = S[to_ind]
        q_el = self.calc_instantaneous_rate(from_state, to_state)
        row[to_ind] = q_el
        row_sum += q_el
      row[from_ind] = -row_sum
    return q
  def state_freqs(self):
    S = self.S
    sf = [0.0]*S.num_states
    for s_ind in xrange(S.num_states):
      state = S[s_ind]
      if state.num_alleles == 1:
        i = state.alleles_list[0]
        sf[s_ind] = self.nuc_freqs[i]*self.f1
        continue
      if state.num_alleles == 2:
        i, j = state.alleles_list
        rf = self.rev_mat[i][j]
        fp = self.nuc_freqs[i]*self.nuc_freqs[j]
        numerator = fp*rf*self.f2
        denominator = self.K2*(2 + self.psi)
        c = numerator/denominator
      elif state.num_alleles == 3:
        i, j, k = state.alleles_list
        rf = sum_of_rate_pair_products(self.rev_mat, i, j, k)
        fp = self.nuc_freqs[i]*self.nuc_freqs[j]*self.nuc_freqs[k]
        numerator = fp*rf*self.f3
        denominator = self.K3*(3 + self.psi)
        c = numerator/denominator
      else:
        assert state.num_alleles == 4
        c = self.f4/(4 + self.psi)
      if state._is_mid_freq:
        sf[s_ind] = self.psi*c
      else:
        sf[s_ind] = c
    return sf
  def prob_mat(self, edge_len):
      q = self.q_mat()
      npq = np.mat(q)
      scaled = edge_len*npq
      return linalg.expm(scaled)
params = {}
params['nuc_freqs'] = [0.1, 0.2, 0.3, 0.4]
params['gtr_rates'] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
params['prob_diallelic'] = 0.01 # Equil. prob. being diallelic
params['psi'] = 0.1 # psi/(2 + psi) = conditional prob of being in 50/50 mid point given dialllelic
params['drift_rate'] = 200.0 # should be >> than GTR rates 
model = GLMHPomoModel(**params)


psf = model.state_freqs()
#print 'state freqs: ', psf
print sum(psf)
q = model.q_mat()
'''for i, row in enumerate(q):
  rs = '   '.join(['{:10.4f}'.format(el) for el in row])
  label = 'Q[{}][*] ='.format(model.S[i])
  print '{:10} {}'.format(label, rs)
  print sum(row)
'''
for i in xrange(model.S.num_states):
  for j in xrange(i + 1, model.S.num_states):
    f = psf[i]*q[i][j]
    r = psf[j]*q[j][i]
    if abs(f - r) > 1e-6:
      print model.S[i], 'freq = ', psf[i], ' q out = ', q[i][j], ' r_out', q[i][j]/psf[j], ' flux_out =', f
      print model.S[j], 'freq = ', psf[j], ' q out = ', q[j][i], ' r_out', q[j][i]/psf[i], ' flux_out =', r
      print model.S[j], psf[i], psf[j], q[i][j], q[j][i], f, r
      print ' diff =', f - r
      sys.exit(1)

edge_len = float(sys.argv[1])
p = model.prob_mat(edge_len)
for i, row in enumerate(p):
  rs = '   '.join(['{:10.4f}'.format(el) for el in row])
  label = 'Pr[{}][*] ='.format(model.S[i])
  print '{:10} {}'.format(label, rs)

"""
class S:
  '''State ordering'''
  A, C, G, T = range(4)
  AH_CL, AM_CM, AL_CH = range(4, 7)  # A-high+C-low, A-mid+C-mid, and A-low+C-high
  AH_GL, AM_GM, AL_GH = range(7, 10) # A-high+G-low, A-mid+G-mid, and A-low+G-high
  AH_TL, AM_TM, AL_TH = range(10, 13)  # A-high+T-low, A-mid+T-mid, and A-low+T-high
  CH_GL, CM_GM, CL_GH = range(13, 16) # C-high+G-low, C-mid+G-mid, and C-low+G-high
  CH_TL, CM_TM, CL_TH = range(16, 19)  # C-high+T-low, C-mid+T-mid, and C-low+T-high
  GH_TL, GM_TM, GL_TH = range(19, 22)  # G-high+T-low, G-mid+T-mid, and G-low+T-high
  AH_CL_GL, AL_CH_GL, AL_CL_GH, AM_CM_GM = range(22, 26)
  AH_CL_TL, AL_CH_TL, AL_CL_TH, AM_CM_TM = range(26, 30)
  AH_GL_TL, AL_GH_TL, AL_GL_TH, AM_GM_TM = range(30, 34)
  CH_GL_TL, CL_GH_TL, CL_GL_TH, CM_GM_TM = range(34, 38)
  AH_CL_GL_TL = 38
  AL_CH_GL_TL = 39
  AL_CL_GH_TL = 40
  AL_CL_GL_TH = 41
  AM_CM_GM_TM = 42

  STATES = [None]*NUM_STATES
  POLY_LOOKUP = {}
  IND_MID =  {AM_CM: [A, C], AM_GM: [A, G], AM_TM: [A, T], CM_GM: [C, G], CM_TM: [C, T], GM_TM: [G, T]}
  IND_LOW =  {AH_CL: [A, C], AH_GL: [A, G], AH_TL: [A, T], CH_GL: [C, G], CH_TL: [C, T], GH_TL: [G, T]}
  IND_HIGH = {AL_CH: [A, C], AL_GH: [A, G], AL_TH: [A, T], CL_GH: [C, G], CL_TH: [C, T], GL_TH: [G, T]}

for state, value in S.__dict__.items():
  if isinstance(value, int):
    S.STATES[value] = state
S.POLY_LOOKUP[S.A] = {S.C: [S.AH_CL, S.AM_CM, S.AL_CH, ],
                      S.G: [S.AH_GL, S.AM_GM, S.AL_GH, ],
                      S.T: [S.AH_TL, S.AM_TM, S.AL_TH, ],
                     }
S.POLY_LOOKUP[S.C] = {S.A: [S.AL_CH, S.AM_CM, S.AH_CL, ],
                      S.G: [S.CH_GL, S.CM_GM, S.CL_GH, ],
                      S.T: [S.CH_TL, S.CM_TM, S.CL_TH, ],
                     }
S.POLY_LOOKUP[S.G] = {S.A: [S.AL_GH, S.AM_GM, S.AH_GL, ],
                      S.C: [S.CL_GH, S.CM_GM, S.CH_GL, ],
                      S.T: [S.GH_TL, S.GM_TM, S.GL_TH, ],
                     }
S.POLY_LOOKUP[S.T] = {S.A: [S.AL_TH, S.AM_TM, S.AH_TL, ],
                      S.C: [S.CL_TH, S.CM_TM, S.CH_TL, ],
                      S.G: [S.GL_TH, S.GM_TM, S.GH_TL, ],
                     }

def lmh_pomo_qmat(params):
  nuc_freqs = params['NUC_FREQ']
  assert min(nuc_freqs) > 0.0
  assert abs(sum(nuc_freqs) - 1) < TOL
  gtr_rates = params['GTR_RATE']
  assert min(gtr_rates) > 0.0
  prob_poly = params['PROB_DIALLELIC']
  assert prob_poly > 0.0
  assert prob_poly < 1.0
  mid_freq_param = params['PSI']
  assert mid_freq_param > 0.0
  drift_rate = params['DRIFT_RATE']
  assert drift_rate > 0.0

  two_plus_psi = 2.0 + mid_freq_param
  freq_for_bins_given_poly = [1/two_plus_psi, mid_freq_param/two_plus_psi, 1/two_plus_psi]

  pi_A, pi_C, pi_G, pi_T = nuc_freqs
  r_AC, r_AG, r_AT, r_CG, r_CT, r_GT = gtr_rates
  
  sym_mu_mat = [[None, r_AC, r_AG, r_AT],
                [r_AC, None, r_CG, r_CT],
                [r_AG, r_CG, None, r_GT],
                [r_AT, r_CT, r_GT, None],
               ]
  raw_mat = [[0.0]*NUM_STATES for i in range(NUM_STATES)]
  mono_over_poly = (1.0 - prob_poly)/prob_poly
  K = 0.0
  for i in range(4):
    for j in range(i + 1, 4):
      pi_prod = nuc_freqs[i]*nuc_freqs[j]
      k_el = pi_prod*sym_mu_mat[i][j]
      K += k_el
  loss_rate = K*mono_over_poly*two_plus_psi
  
  # first the gain of/loss of element
  for mono in [S.A, S.C, S.G, S.T]:
    for other, bins_with_other in S.POLY_LOOKUP[mono].items():
      pi_other = nuc_freqs[other]
      to_other_ind = bins_with_other[0] # first element is the high freq of mono, low of other
      q_new_mut = pi_other*sym_mu_mat[mono][other]
      raw_mat[mono][to_other_ind] = q_new_mut
      raw_mat[to_other_ind][mono] = loss_rate
  # Set the transitions that correspond to changes in allele freq among polymorphic states (not mutation or fixation)
  from_mid_coefficient = prob_poly*drift_rate/(K*two_plus_psi)
  to_mid_coefficient = from_mid_coefficient*mid_freq_param
  for poly_ind in range(4, NUM_STATES):
    if poly_ind in S.IND_MID:
      I, J = S.IND_MID[poly_ind]
      alleles_sp_factor = nuc_freqs[I]*nuc_freqs[J]*sym_mu_mat[I][J]
      q = from_mid_coefficient*alleles_sp_factor
      raw_mat[poly_ind][poly_ind + 1] = q
      raw_mat[poly_ind][poly_ind - 1] = q
    else:
      if poly_ind in S.IND_LOW:
        I, J = S.IND_LOW[poly_ind]
        neighbor = poly_ind + 1
      else:
        assert poly_ind in S.IND_HIGH
        I, J = S.IND_HIGH[poly_ind]
        neighbor = poly_ind - 1
      alleles_sp_factor = nuc_freqs[I]*nuc_freqs[J]*sym_mu_mat[I][J]
      q = to_mid_coefficient * alleles_sp_factor
      raw_mat[poly_ind][neighbor] = q
  # Set diagonal
  for row_ind in range(NUM_STATES):
    row_sum = sum(raw_mat[row_ind])
    assert raw_mat[row_ind][row_ind] == 0.0
    raw_mat[row_ind][row_ind] = -1*row_sum

  # Set scaler so that exit from monomorphic states = 1
  mono_exit_rate = 0.0
  for i in range(4):
    i_exit_rate = 0.0
    for j, r in enumerate(raw_mat[i]):
      if r > 0.0:
        if j in S.IND_LOW:
          I, J = S.IND_LOW[j]
        else:
          assert j in S.IND_HIGH
          I, J = S.IND_HIGH[j]
        assert (I == i) or (J == i)
        alleles_sp_factor = nuc_freqs[I]*nuc_freqs[J]*sym_mu_mat[I][J]
        to_mid_rate = to_mid_coefficient*alleles_sp_factor
        prob_next_move_to_mid = to_mid_rate/(to_mid_rate + loss_rate)
        # once an allele makes to 50/50 freq, its' probability of substitution is 50%
        eventual_subst_prob = 0.5*prob_next_move_to_mid
        i_to_j_subst_rate = eventual_subst_prob*r
        i_exit_rate += i_to_j_subst_rate
    mono_exit_rate += nuc_freqs[i]*i_exit_rate
  mono_exit_rate *= (1.0 - prob_poly)
  scaler = 1.0/mono_exit_rate
  for row in raw_mat:
    for i in range(NUM_STATES):
      row[i] *= scaler
  print 'scaler =', scaler
  return raw_mat



def lmh_pomo_prob(params, edge_len):
  q = lmh_pomo_qmat(params)
  npq = np.mat(q)
  scaled = edge_len*npq
  return linalg.expm(scaled)



q = lmh_pomo_qmat(params)
for i, row in enumerate(q):
  rs = '   '.join(['{:10.4f}'.format(el) for el in row])
  label = 'Q[{}][*] ='.format(i)
  print '{:10} {}'.format(label, rs)

print
import sys
edge_len = float(sys.argv[1])
p = lmh_pomo_prob(params, edge_len)
for i, row in enumerate(p):
  rs = '   '.join(['{:10.4f}'.format(el) for el in row])
  label = 'P[{}][*] ='.format(i)
  print '{:10} {}'.format(label, rs)
"""