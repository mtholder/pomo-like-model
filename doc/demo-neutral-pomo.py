import numpy as np
from scipy import linalg
import sys
TOL = 1e-8
class NUC:
  A, C, G, T = range(4)
class GTR_RATE:
  AC, AG, AT, CG, CT, GT = range(6)

try:
    VIRTUAL_POP_SIZE = int(sys.argv[1])
    assert VIRTUAL_POP_SIZE > 1
    NUM_POLY_BINS_PER_DIALLELE = VIRTUAL_POP_SIZE - 1
except:
    sys.exit('Expecting the first argument to be the virtual population size (must be > 1)')
try:
    edge_len = float(sys.argv[2])
    assert edge_len >= 0.0
except:
    sys.exit('Expecting the second argument to be an edge length (must be >= 0.0)')

NUM_POMO_STATES = 4 + 6 * NUM_POLY_BINS_PER_DIALLELE

class DIALLELIC_PAIRS:
  ORDERING = ('AC', 'AG', 'AT', 'CG', 'CT', 'GT')
  TO_CLASS_INDEX = {'AC': 0, 'AG': 1, 'AT': 2, 'CG': 3, 'CT': 4, 'GT': 5}

class S:
    '''State ordering'''
    A, C, G, T = range(4)
    STATES = [None]*NUM_POMO_STATES
    POLY_LOOKUP = {}
    @staticmethod
    def is_monomorphic(i):
        return i < 4
    @staticmethod
    def diallele_category(i):
        ofs = i - 4 # subtract off the 4 mono states
        return ofs // NUM_POLY_BINS_PER_DIALLELE
    @staticmethod
    def diallele_pair_code_to_single_codes(diallele_pair_code):
        f, s = DIALLELIC_PAIRS.ORDERING[diallele_pair_code]
        return S.STATES.index(f), S.STATES.index(s)
    
    @staticmethod
    def diallele_count(di_state, diallele_pair_code, single):
        diallele_letters = DIALLELIC_PAIRS.ORDERING[diallele_pair_code]
        single_letter = S.STATES[single]
        ofs = (di_state - 4) % (VIRTUAL_POP_SIZE - 1)
        num_second = 1 + ofs
        num_first = VIRTUAL_POP_SIZE - num_second
        if single_letter == diallele_letters[0]:
            return num_first
        elif single_letter == diallele_letters[1]:
            return num_second
        return 0
    @staticmethod
    def other_allele_letter(diallele_pair_code, single):
        diallele_letters = DIALLELIC_PAIRS.ORDERING[diallele_pair_code]
        single_letter = S.STATES[single]
        if single_letter == diallele_letters[0]:
            return diallele_letters[1]
        else:
            assert single_letter == diallele_letters[1]
            return diallele_letters[0]
    @staticmethod
    def other_allele(diallele_pair_code, single):
        l = S.other_allele_letter(diallele_pair_code, single)
        return 'ACGT'.index(l)

for nuc in 'ACGT':
    S.STATES[S.__dict__[nuc]] = nuc

for pair, value in DIALLELIC_PAIRS.TO_CLASS_INDEX.items():
    offset = 4 - 1 + value*NUM_POLY_BINS_PER_DIALLELE
    print 'pv', pair, value, 'offset=', offset
    for s in range(1, VIRTUAL_POP_SIZE):
        f = VIRTUAL_POP_SIZE - s
        name = pair[0] + str(f) + pair[1] + str(s)
        s_index = offset + s
        print s_index, name
        S.STATES[s_index] = name

print '\n'.join(['{} {}'.format(i, s) for i, s in enumerate(S.STATES)])

params = {}
params['NUC_FREQ'] = [0.1, 0.2, 0.3, 0.4]
params['GTR_RATE'] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
params['PROB_POLY'] = 0.1 # Equil. prob. being polymorphic

def calc_K(virtual_pop_size, nuc_freqs, sym_mu_mat):
    K = virtual_pop_size
    s = 0.0
    for i in range(1, virtual_pop_size):
        s += 1.0 / float(i*(virtual_pop_size - i))
    K *= s
    f = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            f += nuc_freqs[i]*nuc_freqs[j]*sym_mu_mat[i][j]
    K *= f
    return K

def set_q_diagonal(q):
    for i in range(len(q)):
        q[i][i] = 0.0
        row_sum = sum(q[i])
        q[i][i] = -row_sum

def calc_state_freq(K, prob_poly, virtual_pop_size, nuc_freqs, sym_mu_mat):
    sf = [0.0] * NUM_POMO_STATES
    poly_coeff = prob_poly*virtual_pop_size/K
    for i in xrange(NUM_POMO_STATES):
        if S.is_monomorphic(i):
            sf[i] = nuc_freqs[i]*(1 - prob_poly)
        else:
            i_diallele = S.diallele_category(i)
            pc = S.diallele_pair_code_to_single_codes(i_diallele)
            f, s = pc
            i_count_in_i = S.diallele_count(i, i_diallele, f)
            nmi = virtual_pop_size - i_count_in_i
            assert nmi > 0
            assert i_count_in_i > 0
            denom = nmi * i_count_in_i
            sf[i] = poly_coeff * nuc_freqs[f] * nuc_freqs[s] * sym_mu_mat[f][s]/denom
    tp = sum(sf)
    print 
    assert abs(tp - 1.0) < 1e-8
    return sf

def check_time_reversibility(q, state_freq):
    assert len(q) == len(state_freq)
    for i in range(len(q)):
        for j in range(len(q)):
            fwd = state_freq[i]*q[i][j]
            rev = state_freq[j]*q[j][i]
            if abs(fwd - rev) > 1e-7:
                print 'state_freq[{i}]*q[{i}][{j}] != state_freq[{j}]*q[{j}][{i}]'.format(i=i, j=j)
                print '{f} * {q} != {r} * {z}'.format(f=state_freq[i], q=q[i][j], r=state_freq[j], z=q[j][i])
                print '{f} != {r}'.format(f=fwd, r=rev)
                assert abs(fwd - rev) < 1e-7

def neut_pomo_qmat(params):
    nuc_freqs = params['NUC_FREQ']
    assert min(nuc_freqs) > 0.0
    assert abs(sum(nuc_freqs) - 1) < TOL
    gtr_rates = params['GTR_RATE']
    assert min(gtr_rates) > 0.0
    prob_poly = params['PROB_POLY']
    assert prob_poly > 0.0
    assert prob_poly < 1.0
    poly_transform = (1 - prob_poly)/prob_poly
    pi_A, pi_C, pi_G, pi_T = nuc_freqs
    r_AC, r_AG, r_AT, r_CG, r_CT, r_GT = gtr_rates
    sym_mu_mat = [[None, r_AC, r_AG, r_AT],
                [r_AC, None, r_CG, r_CT],
                [r_AG, r_CG, None, r_GT],
                [r_AT, r_CT, r_GT, None],
               ]
    K = calc_K(VIRTUAL_POP_SIZE, nuc_freqs, sym_mu_mat)
    q = [[0]* NUM_POMO_STATES for i in range(NUM_POMO_STATES)]
    for i in range(NUM_POMO_STATES):
        i_is_mono = S.is_monomorphic(i)
        if not i_is_mono:
            i_diallele = S.diallele_category(i)
        for j in range(NUM_POMO_STATES):
            if i == j:
                continue # diagonal is handled in the loop below
            q_el = 0.0
            j_is_mono = S.is_monomorphic(j)
            if not j_is_mono:
                j_diallele = S.diallele_category(j)
            if i_is_mono:
                if not j_is_mono:
                    i_count_in_j = S.diallele_count(j, j_diallele, i)
                    #print 'i,j,i_count_in_j,j_diallele = ', i, j, i_count_in_j, j_diallele
                    if i_count_in_j == NUM_POLY_BINS_PER_DIALLELE:
                        # i-> j is new mutation (eqn 18)
                        mut = S.other_allele(j_diallele, i)
                        r = sym_mu_mat[i][mut]
                        f = nuc_freqs[mut]
                        q_el = VIRTUAL_POP_SIZE*VIRTUAL_POP_SIZE*r*f
            else:
                if j_is_mono:
                    j_count_in_i = S.diallele_count(i, i_diallele, j)
                    if j_count_in_i == 1:
                        # i -> j is a loss of an allelle (eqn 21)
                        q_el = K * poly_transform*VIRTUAL_POP_SIZE*(VIRTUAL_POP_SIZE - 1)
                elif j_diallele == i_diallele:
                    pc = S.diallele_pair_code_to_single_codes(i_diallele)
                    f = pc[0]
                    i_count_in_i = S.diallele_count(i, i_diallele, f)
                    i_count_in_j = S.diallele_count(j, i_diallele, f)
                    diff = i_count_in_i - i_count_in_j
                    if (diff == 1) or (diff == -1):
                        # drift. eqn 14
                        q_el = i_count_in_i * (VIRTUAL_POP_SIZE - i_count_in_i)/float(VIRTUAL_POP_SIZE)
            q[i][j] = q_el
    set_q_diagonal(q)
    state_freq = calc_state_freq(K, prob_poly, VIRTUAL_POP_SIZE, nuc_freqs, sym_mu_mat)
    check_time_reversibility(q, state_freq)
    return q

def neut_pomo_prob(params, edge_len):
  q = neut_pomo_qmat(params)
  npq = np.mat(q)
  scaled = edge_len*npq
  return linalg.expm(scaled)

q = neut_pomo_qmat(params)
for i, row in enumerate(q):
  rs = '   '.join(['{:10.4f}'.format(el) for el in row])
  label = 'Q[{} ({})][*] ='.format(i, S.STATES[i]).ljust(16)
  print '{:10} {}'.format(label, rs)

p = neut_pomo_prob(params, edge_len)
for i, row in enumerate(p):
  rs = '   '.join(['{:10.4f}'.format(el) for el in row])
  label = 'P[{}][*] ='.format(i).ljust(16)
  print '{:10} {}'.format(label, rs)
