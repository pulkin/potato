import numpy as np

from pyscf.pbc.cc.kccsd_rhf import nested_to_vector, padding_k_idx
from pyscf.pbc.lib import kpts_helper

import greens_function as gf

###################
# EA Greens       #
###################


class PaddingError(AssertionError):
    pass


def assert_non_padded(cc, p, kp):
    if p not in padding_k_idx(cc, kind="joint")[kp]:
        raise PaddingError("The index p={:d} at k={:d} is padded".format(p, kp))


def greens_b_singles_ea_krhf(t1, p, kp):
    return gf.greens_b_singles_ea_rhf(t1[kp], p)


def greens_b_doubles_ea_krhf(t2, p, kp):
    result = []
    nkpts = len(t2)
    for ki in range(nkpts):
        result_k = []
        result.append(result_k)
        for ka in range(nkpts):
            result_k.append(gf.greens_b_doubles_ea_rhf(t2[kp][ki][ka], p))
    return result


def greens_b_vector_ea_krhf(cc, p, kp):
    assert_non_padded(cc, p, kp)
    return nested_to_vector((
        greens_b_singles_ea_krhf(cc.t1, p, kp),
        greens_b_doubles_ea_krhf(cc.t2, p, kp),
    ))[0]


def greens_e_singles_ea_krhf(t1, t2, l1, l2, p, kp, kconserv):
    # Note that nocc, nvir are those at a specific k-point
    nocc, nvir = t1[kp].shape
    ds_type = t1.dtype
    nkpts = len(t1)
    if p < nocc:
        return l1[kp, p, :]
    else:
        result = np.zeros((nvir,), dtype=ds_type)
        result[p - nocc] = -1.0
        result += np.einsum('ia,i->a', l1[kp], t1[kp][:, p - nocc])
        for kk in range(nkpts):
            for kl in range(nkpts):
                #   OLD: ka = kconserv[kl, kp, kk]
                kc = kconserv[kl, kp, kk]
                # NON-K: result += 2 * np.einsum('klca,klc->a', l2, t2[:, :, :, p - nocc])
                #   OLD: result += 2 * np.einsum('klca,klc->a', l2[ka][kk][kl], t2[ka][kk][kl][:, :, :, p - nocc])
                result += 2 * np.einsum('klca,klc->a', l2[kk][kl][kc], t2[kk][kl][kc][:, :, :, p - nocc])
                # NON-K: result -= np.einsum('klca,lkc->a', l2, t2[:, :, :, p - nocc])
                #   OLD: result -= np.einsum('klca,lkc->a', l2[ka][kk][kl], t2[kk][ka][kl][:, :, :, p - nocc])
                result -= np.einsum('klca,lkc->a', l2[kk][kl][kc], t2[kl][kk][kc][:, :, :, p - nocc])
        return result


def greens_e_doubles_ea_krhf(t1, l1, l2, p, kp, kconserv):
    # Note that nocc, nvir are those at a specific k-point
    nocc, nvir = t1[kp].shape
    nkpts = len(t1)
    if p < nocc:

        # NON-K: 2 * l2[p, :, :, :] - l2[:, p, :, :]
        result = []

        for ki in range(nkpts):
            result_k = []
            # OLD: for kj in range(nkpts):
            #    OLD: result_k.append(2 * l2[ki][kj][kp][p, :, :, :] - l2[kj][ki][kp][:, p, :, :])
            for ka in range(nkpts):
                result_k.append(2 * l2[kp][ki][ka][p, :, :, :] - l2[ki][kp][ka][:, p, :, :])
            result.append(result_k)

        return result

    else:

        result = []

        # NON-K: result += 2 * np.einsum('k,jkba->jab', t1[:, p - nocc], l2)
        # NON-K: result -= np.einsum('k,jkab->jab', t1[:, p - nocc], l2)
        for kj in range(nkpts):
            result_k = []
            for ka in range(nkpts):
                kb = kconserv[kp, kj, ka]
                result_k.append(2 * np.einsum('k,jkba->jab', t1[kp][:, p - nocc], l2[kj][kp][kb]) -
                                np.einsum('k,jkab->jab', t1[kp][:, p - nocc], l2[kj][kp][ka]))
            result.append(result_k)

        # NON-K: result[:, p - nocc, :] += -2. * l1
        # OLD: for kb in range(nkpts):
            # OLD: result[kb][kp][:, p - nocc, :] += -2 * l1[kb]
        for ki in range(nkpts):
            result[ki][kp][:, p - nocc, :] += -2 * l1[ki]

        # NON-K: result[:, :, p - nocc] += l1
        # OLD: for ka in range(nkpts):
            # OLD: result[ka][ka][:, :, p - nocc] += l1[ka]
        for ki in range(nkpts):
            result[ki][ki][:, :, p - nocc] += l1[ki]

        return result


def greens_e_vector_ea_krhf(cc, p, kp):
    assert_non_padded(cc, p, kp)
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    return nested_to_vector((
        greens_e_singles_ea_krhf(cc.t1, cc.t2, cc.l1, cc.l2, p, kp, kconserv),
        greens_e_doubles_ea_krhf(cc.t1, cc.l1, cc.l2, p, kp, kconserv),
    ))[0]


###################
# IP Greens       #
###################


def greens_b_singles_ip_krhf(t1, p, kp):
    # NON-K: return t1[:, p - nocc]
    return gf.greens_b_singles_ip_rhf(t1[kp], p)


def greens_b_doubles_ip_krhf(t2, p, kp, kconserv):
    result = []
    nkpts = len(t2)
    for ki in range(nkpts):
        result_k = []
        result.append(result_k)
        for kj in range(nkpts):
            ka = kconserv[ki, kp, kj]
            # NON-K: return t2[:, :, :, p - nocc]
            result_k.append(gf.greens_b_doubles_ip_rhf(t2[ki][kj][ka], p))
    return result


def greens_b_vector_ip_krhf(cc, p, kp):
    assert_non_padded(cc, p, kp)
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    return nested_to_vector((
        greens_b_singles_ip_krhf(cc.t1, p, kp),
        greens_b_doubles_ip_krhf(cc.t2, p, kp, kconserv),
    ))[0]


def greens_e_singles_ip_krhf(t1, t2, l1, l2, p, kp, kconserv):
    # Note that nocc, nvir are those at a specific k-point
    nocc, nvir = t1[kp].shape
    ds_type = t1.dtype
    nkpts = len(t1)
    if p >= nocc:

        # NON-K: return -l1[:, p - nocc]
        return -l1[kp][:, p - nocc]

    else:

        result = np.zeros((nocc,), dtype=ds_type)
        result[p] = -1.0

        # NON-K: result += np.einsum('ia,a->i', l1, t1[p, :])
        result += np.einsum('ia,a->i', l1[kp], t1[kp][p, :])

        for kl in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[kp][kl][kc]
                # NON-K: result += 2 * np.einsum('ilcd,lcd->i', l2, t2[p, :, :, :])
                result += 2 * np.einsum('ilcd,lcd->i', l2[kp][kl][kc], t2[kp][kl][kc][p, :, :, :])
                # NON-K: result -= np.einsum('ilcd,ldc->i', l2, t2[p, :, :, :])
                result -= np.einsum('ilcd,ldc->i', l2[kp][kl][kc], t2[kp][kl][kd][p, :, :, :])

        return result


def greens_e_doubles_ip_krhf(t1, l1, l2, p, kp, kconserv):
    # Note that nocc, nvir are those at a specific k-point
    nocc, nvir = t1[kp].shape
    nkpts = len(t1)
    if p >= nocc:

        result = []

        for ki in range(nkpts):
            result_k = []
            for kj in range(nkpts):
                # NON-K: return -2 * l2[:, :, p - nocc, :] + l2[:, :, :, p - nocc]
                kb = kconserv[ki, kj, kp]
                result_k.append(-2 * l2[ki][kj][kp][:, :, p - nocc, :] + l2[ki][kj][kb][:, :, :, p - nocc])
            result.append(result_k)

        return result

    else:

        result = []

        for ki in range(nkpts):
            result_k = []
            for kj in range(nkpts):
                # NON-K: result += 2 * np.einsum('c,ijcb->ijb', t1[p, :], l2)
                # NON-K: result -= np.einsum('c,jicb->ijb', t1[p, :], l2)
                result_k.append(2 * np.einsum('c,ijcb->ijb', t1[kp][p, :], l2[ki][kj][kp]) -
                                np.einsum('c,jicb->ijb', t1[kp][p, :], l2[kj][ki][kp]))
            result.append(result_k)

        # result[p, :, :] += -2. * l1
        for kj in range(nkpts):
            result[kp][kj][p, :, :] += -2 * l1[kj]

        # result[:, p, :] += l1
        for ki in range(nkpts):
            result[ki][kp][:, p, :] += l1[ki]

        return result


def greens_e_vector_ip_krhf(cc, p, kp):
    assert_non_padded(cc, p, kp)
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    return nested_to_vector((
        greens_e_singles_ip_krhf(cc.t1, cc.t2, cc.l1, cc.l2, p, kp, kconserv),
        greens_e_doubles_ip_krhf(cc.t1, cc.l1, cc.l2, p, kp, kconserv),
    ))[0]


###################
# Solvers         #
###################


class EOMIPGFatK(gf.EOMIPGF):
    def __init__(self, cc, k):
        """IP routines: matvec, e and b."""
        gf.EOMGF.__init__(self, cc)
        self.diagonal = cc.ipccsd_diag(k)
        self.k = k

    def make_e(self, p):
        return greens_e_vector_ip_krhf(self.cc, p, self.k)

    def make_b(self, q):
        return greens_b_vector_ip_krhf(self.cc, q, self.k)

    def matvec(self, vec, energy=None):
        if energy is None:
            return self.cc.ipccsd_matvec(vec, self.k)
        else:
            return self.cc.ipccsd_matvec(vec, self.k) + vec * energy

    def iguess(self, q, energy):
        return self.make_b(q) / (energy - self.cc.eris.fock[self.k][q, q])


class EOMEAGFatK(gf.EOMEAGF):
    def __init__(self, cc, k):
        """EA routines: matvec, e and b."""
        gf.EOMGF.__init__(self, cc)
        self.diagonal = cc.eaccsd_diag(k)
        self.k = k

    def make_e(self, p):
        return greens_e_vector_ea_krhf(self.cc, p, self.k)

    def make_b(self, q):
        return greens_b_vector_ea_krhf(self.cc, q, self.k)

    def matvec(self, vec, energy=None):
        if energy is None:
            return -self.cc.eaccsd_matvec(vec, self.k)
        else:
            return -self.cc.eaccsd_matvec(vec, self.k) + vec * energy

    def iguess(self, q, energy):
        return self.make_b(q) / (energy - self.cc.eris.fock[self.k][q, q])


def k_interfaces(cls, cc, kpts):
    """Builds a list of k-dependent interfaces."""
    return list(cls(cc, k) for k in kpts)


def gf_k(interfaces, parent, ps, qs, kwargs, return_ncalls=False):
    """K-dependent Green's function."""
    ncalls = 0
    result_gf = []
    log = interfaces[0].logger
    log.note("Starting a K-GF calculation n_k = {:d}".format(
        len(interfaces),
    ))
    ps = np.array(ps)
    qs = np.array(qs)
    for interface in interfaces:
        log.note("Running K-GF @ id(k)={:d} ...".format(len(result_gf)))
        idx = padding_k_idx(interface.cc, kind="joint")[interface.k]
        ps_physical = np.any(ps[:, np.newaxis] == idx[np.newaxis, :], axis=1)
        qs_physical = np.any(qs[:, np.newaxis] == idx[np.newaxis, :], axis=1)
        kwargs.update(dict(
            return_ncalls=return_ncalls,
            interface=interface,
            ps=ps[ps_physical],
            qs=qs[qs_physical],
        ))
        result = parent(**kwargs)
        if return_ncalls:
            ncalls += result[1]
            result = result[0]
        result_gf.append(result)

    gfvals = np.array(result_gf)

    if return_ncalls:
        return gfvals, ncalls
    else:
        return gfvals


def gf_time(interfaces, ps, qs, times, imaginary=False, return_ncalls=False, energy_shift=0, **kwargs):
    """Time-dependent Green's function."""
    return gf_k(
        interfaces,
        gf.gf_time,
        ps, qs,
        dict(times=times, imaginary=imaginary, energy_shift=energy_shift, **kwargs),
        return_ncalls=return_ncalls,
    )


def gf_fourier(interfaces, ps, qs, energies, sampling_factor=None, return_ncalls=False):
    """Fourier-transformed energy-dependent GF."""
    return gf_k(
        interfaces,
        gf.gf_fourier,
        ps, qs,
        dict(energies=energies, sampling_factor=sampling_factor),
        return_ncalls=return_ncalls,
    )


def gf_chebushev(interfaces, ps, qs, energies, order, a, return_ncalls=False):
    """Chebushev expansion of the GF by Wolf et al. PRB 90, 115124 (2014)."""
    return gf_k(
        interfaces,
        gf.gf_chebushev,
        ps, qs,
        dict(energies=energies, order=order, a=a),
        return_ncalls=return_ncalls,
    )


def gf_iter(interfaces, ps, qs, energies, return_ncalls=False, **kwargs):
    """Conventional GF iterations."""
    return gf_k(
        interfaces,
        gf.gf_iter,
        ps, qs,
        dict(energies=energies, **kwargs),
        return_ncalls=return_ncalls,
    )
