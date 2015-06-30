import numpy as np
from scipy import linalg
TOL = 1e-8
NUM_STATES = 22
class NUC:
  A, C, G, T = range(4)
class GTR_RATE:
  AC, AG, AT, CG, CT, GT = range(6)

class S:
  '''State ordering'''
  A, C, G, T = range(4)
  AH_CL, AM_CM, AL_CH = range(4, 7)  # A-high+C-low, A-mid+C-mid, and A-low+C-high
  AH_GL, AM_GM, AL_GH = range(7, 10) # A-high+G-low, A-mid+G-mid, and A-low+G-high
  AH_TL, AM_TM, AL_TH = range(10, 13)  # A-high+T-low, A-mid+T-mid, and A-low+T-high
  CH_GL, CM_GM, CL_GH = range(13, 16) # C-high+G-low, C-mid+G-mid, and C-low+G-high
  CH_TL, CM_TM, CL_TH = range(16, 19)  # C-high+T-low, C-mid+T-mid, and C-low+T-high
  GH_TL, GM_TM, GL_TH = range(19, 22)  # G-high+T-low, G-mid+T-mid, and G-low+T-high
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
  prob_poly = params['PROB_POLY']
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


params = {}
params['NUC_FREQ'] = [0.1, 0.2, 0.3, 0.4]
params['GTR_RATE'] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
params['PROB_POLY'] = 0.01 # Equil. prob. being polymorphic
params['PSI'] = 0.1 # psi/(2 + psi) = conditional prob of being in 50/50 mid point given polymorphic
params['DRIFT_RATE'] = 200.0 # should be >> than GTR rates 

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
