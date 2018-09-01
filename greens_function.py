import numpy as np
from numpy.polynomial import chebyshev
import scipy
from scipy import integrate
from scipy.sparse import linalg as spla

import pyscf
import pyscf.cc
from pyscf.cc.eom_rccsd import amplitudes_to_vector_ip, amplitudes_to_vector_ea, ipccsd_diag, eaccsd_diag
from pyscf.lib import logger


###################
# EA Greens       #
###################


def greens_b_singles_ea_rhf(t1, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        return -t1[p, :]
    else:
        result = np.zeros((nvir,), dtype=ds_type)
        result[p - nocc] = 1.0
        return result


def greens_b_doubles_ea_rhf(t2, p):
    nocc, _, nvir, _ = t2.shape
    ds_type = t2.dtype
    if p < nocc:
        return -t2[p, :, :, :]
    else:
        return np.zeros((nocc, nvir, nvir), dtype=ds_type)


def greens_b_vector_ea_rhf(cc, p):
    return amplitudes_to_vector_ea(
        greens_b_singles_ea_rhf(cc.t1, p),
        greens_b_doubles_ea_rhf(cc.t2, p),
    )


def greens_e_singles_ea_rhf(t1, t2, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        return l1[p, :]
    else:
        result = np.zeros((nvir,), dtype=ds_type)
        result[p - nocc] = -1.0
        result += np.einsum('ia,i->a', l1, t1[:, p - nocc])
        result += 2 * np.einsum('klca,klc->a', l2, t2[:, :, :, p - nocc])
        result -= np.einsum('klca,lkc->a', l2, t2[:, :, :, p - nocc])
        return result


def greens_e_doubles_ea_rhf(t1, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        return 2 * l2[p, :, :, :] - l2[:, p, :, :]
    else:
        result = np.zeros((nocc, nvir, nvir), dtype=ds_type)
        result[:, p - nocc, :] += -2. * l1
        result[:, :, p - nocc] += l1
        result += 2 * np.einsum('k,jkba->jab', t1[:, p - nocc], l2)
        result -= np.einsum('k,jkab->jab', t1[:, p - nocc], l2)
        return result


def greens_e_vector_ea_rhf(cc, p):
    return amplitudes_to_vector_ea(
        greens_e_singles_ea_rhf(cc.t1, cc.t2, cc.l1, cc.l2, p),
        greens_e_doubles_ea_rhf(cc.t1, cc.l1, cc.l2, p),
    )


def ea_size(cc):
    nocc, nvir = cc.t1.shape
    return nvir + nocc * nvir * nvir


def initial_ea_guess(cc):
    return np.zeros(ea_size(cc), dtype=complex)


###################
# IP Greens       #
###################


def greens_b_singles_ip_rhf(t1, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        result = np.zeros((nocc,), dtype=ds_type)
        result[p] = 1.0
        return result
    else:
        return t1[:, p - nocc]


def greens_b_doubles_ip_rhf(t2, p):
    nocc, _, nvir, _ = t2.shape
    ds_type = t2.dtype
    if p < nocc:
        return np.zeros((nocc, nocc, nvir), dtype=ds_type)
    else:
        return t2[:, :, :, p - nocc]


def greens_b_vector_ip_rhf(cc, p):
    return amplitudes_to_vector_ip(
        greens_b_singles_ip_rhf(cc.t1, p),
        greens_b_doubles_ip_rhf(cc.t2, p),
    )


def greens_e_singles_ip_rhf(t1, t2, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        result = np.zeros((nocc,), dtype=ds_type)
        result[p] = -1.0
        result += np.einsum('ia,a->i', l1, t1[p, :])
        result += 2 * np.einsum('ilcd,lcd->i', l2, t2[p, :, :, :])
        result -= np.einsum('ilcd,ldc->i', l2, t2[p, :, :, :])
        return result
    else:
        return -l1[:, p - nocc]


def greens_e_doubles_ip_rhf(t1, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        result = np.zeros((nocc, nocc, nvir), dtype=ds_type)
        result[p, :, :] += -2. * l1
        result[:, p, :] += l1
        result += 2 * np.einsum('c,ijcb->ijb', t1[p, :], l2)
        result -= np.einsum('c,jicb->ijb', t1[p, :], l2)
        return result
    else:
        return -2 * l2[:, :, p - nocc, :] + l2[:, :, :, p - nocc]


def greens_e_vector_ip_rhf(cc, p):
    return amplitudes_to_vector_ip(
        greens_e_singles_ip_rhf(cc.t1, cc.t2, cc.l1, cc.l2, p),
        greens_e_doubles_ip_rhf(cc.t1, cc.l1, cc.l2, p),
    )


def ip_size(cc):
    nocc, nvir = cc.t1.shape
    return nocc + nocc * nocc * nvir


def initial_ip_guess(cc):
    return np.zeros(ip_size(cc), dtype=complex)


###################
# Solvers         #
###################


class CallCounter(object):
    def __init__(self, callable):
        """Counts iterations."""
        self.callable = callable
        self.n_calls = 0

    def __call__(self, *args, **kwargs):
        self.n_calls += 1
        return self.callable(*args, **kwargs)


def gf_retarded_convention(energies, broadening):
    """Conventional energy with an imaginary part (retarded GF)."""
    return np.array(energies).real - 1.j * abs(broadening)


class EOMGF(object):
    """An interface to pyscf. This class is here mostly because of k-gf implementation."""
    def __init__(self, cc):
        self.cc = cc
        self.logger = logger.Logger(cc.stdout, cc.verbose)
        self.vec_log = []
        self.res_log = []
        self.__log_rhs__ = 0

    def make_e(self, p):
        raise NotImplementedError

    def make_b(self, q):
        raise NotImplementedError

    def matvec(self, vec, energy=None):
        raise NotImplementedError

    def precond(self, energy):
        raise NotImplementedError

    def iguess(self, q, energy):
        raise NotImplementedError

    def matvec_logging(self, vec, energy=None):
        self.vec_log.append(np.array(vec))
        result = self.matvec(vec, energy)
        self.res_log.append((abs(result - self.__log_rhs__)**2).sum()**.5)
        return result

    def plot_log(self):
        from matplotlib import pyplot
        vec_log = np.array(self.vec_log)
        vec_log = abs(vec_log[1:] - vec_log[:-1])
        print self.res_log
        fig, (a1, a2) = pyplot.subplots(2, sharex=True, gridspec_kw=dict(
            hspace=0,
        ))
        a1.matshow(vec_log.T, aspect='auto')
        a2.semilogy(np.arange(len(self.res_log)), self.res_log)
        a2.set_xlabel("iter step")
        a2.set_xlim(0, len(self.res_log) - 1)
        pyplot.show()

    def reset_log(self, rhs=0):
        self.vec_log = []
        self.res_log = []
        self.__log_rhs__ = rhs


class EOMIPGF(EOMGF):
    def __init__(self, cc):
        """IP routines: matvec, e and b."""
        super(EOMIPGF, self).__init__(cc)
        self.eomcc = pyscf.cc.eom_rccsd.EOMIP(cc)
        self.imds = self.eomcc.make_imds()
        self.diagonal = ipccsd_diag(self.eomcc, self.imds)

    def make_e(self, p):
        return greens_e_vector_ip_rhf(self.cc, p)

    def make_b(self, q):
        return greens_b_vector_ip_rhf(self.cc, q)

    def matvec(self, vec, energy=None):
        if energy is None:
            return self.eomcc.matvec(vec, imds=self.imds)
        else:
            return self.eomcc.matvec(vec, imds=self.imds) + vec * energy

    def precond(self, energy):
        denom = energy + self.diagonal
        self.logger.debug2("precond denominator min @E={:.5f}: {:.3e} (real: {:.3e})".format(
            energy, min(abs(denom)), min(abs(denom.real)))
        )
        return lambda vector: vector / denom

    def iguess(self, q, energy):
        return self.make_b(q) / (energy - self.imds.eris.fock[q, q])


class EOMEAGF(EOMGF):
    def __init__(self, cc):
        """IP routines: matvec, e and b."""
        super(EOMEAGF, self).__init__(cc)
        self.eomcc = pyscf.cc.eom_rccsd.EOMEA(cc)
        self.imds = self.eomcc.make_imds()
        self.diagonal = eaccsd_diag(self.eomcc, self.imds)

    def make_e(self, p):
        return greens_e_vector_ea_rhf(self.cc, p)

    def make_b(self, q):
        return greens_b_vector_ea_rhf(self.cc, q)

    def matvec(self, vec, energy=None):
        if energy is None:
            return -self.eomcc.matvec(vec, imds=self.imds)
        else:
            return -self.eomcc.matvec(vec, imds=self.imds) + vec * energy

    def precond(self, energy):
        denom = energy - self.diagonal
        self.logger.debug2("precond denominator min @E={:.5f}: {:.3e} (real: {:.3e})".format(
            energy, min(abs(denom)), min(abs(denom.real)))
        )
        return lambda vector: vector / denom

    def iguess(self, q, energy):
        return self.make_b(q) / (energy - self.imds.eris.fock[q, q])


def propagate(initial_state, matvec, times, imaginary=False, return_ncalls=False, **kwargs):
    """Propagates the vector."""

    if imaginary:
        time_unit = 1
    else:
        time_unit = 1j

    ti = times[0]
    tf = times[-1]

    initial_state = np.asarray(initial_state, dtype=np.complex128)

    def matr_multiply(t, vector, args=None):
        return time_unit * matvec(vector)

    matr_multiply = CallCounter(matr_multiply)
    solp = scipy.integrate.solve_ivp(matr_multiply, (ti, tf), initial_state, t_eval=times, **kwargs)
    if return_ncalls:
        return solp.y, matr_multiply.n_calls
    else:
        return solp.y


def gf_time(interface, ps, qs, times, imaginary=False, return_ncalls=False, energy_shift=0, **kwargs):
    """Time-dependent Green's function."""
    if imaginary:
        dtype = np.float64
    else:
        dtype = np.complex128

    gfvals = np.zeros((len(qs), len(ps), len(times)), dtype=dtype)

    # Kets
    e_vector = list()
    for q in qs:
        e_vector.append(interface.make_e(q))

    def matr_multiply(vector):
        return interface.matvec(vector) - energy_shift * vector

    matr_multiply = CallCounter(matr_multiply)
    for ip, p in enumerate(ps):

        # Bra
        b_vector = interface.make_b(p)

        vecs = propagate(
            b_vector,
            matr_multiply,
            times,
            imaginary=imaginary,
            **kwargs
        )

        for iq in range(len(qs)):
            gfvals[iq, ip, :] = np.dot(e_vector[iq], vecs)

    if not imaginary:
        gfvals *= 1.j

    if return_ncalls:
        return gfvals, matr_multiply.n_calls
    else:
        return gfvals


def gf_fourier(interface, ps, qs, energies, sampling_factor=None, return_ncalls=False):
    """Fourier-transformed energy-dependent GF."""
    if sampling_factor is None:
        sampling_factor = 1
    energies = np.array(energies)
    omega_middle = .5 * (min(energies.real) + max(energies.real))
    de = (energies[1] - energies[0]).real
    ntimes = int(2 ** np.ceil(np.log2(sampling_factor * len(energies)))) + 1
    times = np.linspace(0, sampling_factor / de, ntimes)
    # Enforce img part positive
    energies = energies.real + 1.j * abs(energies.imag)
    weights = np.exp(1.j * times[:, np.newaxis] * (energies + omega_middle)[np.newaxis, :])
    if return_ncalls:
        tdgf, ncalls = gf_time(interface, ps, qs, times, energy_shift=omega_middle, return_ncalls=True)
    else:
        tdgf = gf_time(interface, ps, qs, times, energy_shift=omega_middle, return_ncalls=False)
    integrand = tdgf[..., np.newaxis] * weights[(np.newaxis,) * (tdgf.ndim-1) + (slice(None), slice(None))]
    result = integrate.romb(
        integrand,
        dx=times[1] - times[0],
        axis=-2,
    )
    if return_ncalls:
        return result, ncalls
    else:
        return result


def gf_chebushev(interface, ps, qs, energies, order, a, return_ncalls=False):
    """Chebushev expansion of the GF by Wolf et al. PRB 90, 115124 (2014)."""
    e_vector = list()
    for q in qs:
        e_vector.append(interface.make_e(q))

    # Calculate <-|a H^n a+|->
    mu = np.zeros((len(ps), len(qs), order))
    matvec_callable = CallCounter(interface.matvec)
    for ip, p in enumerate(ps):
        b_vector = prev = None
        for o in range(order):
            if o == 0:
                b_vector = interface.make_b(p)
            elif o == 1:
                prev = b_vector
                b_vector = np.array(matvec_callable(b_vector)) / a
            else:
                prev, b_vector = b_vector, 2 * np.array(matvec_callable(b_vector)) / a - prev
            for iq, q in enumerate(qs):
                # Overlap <-| a |psi>
                mu[iq, ip, o] = np.dot(e_vector[iq], b_vector)

    def weight(x, _order=1):
        return (1. if _order == 0 else 2.) / np.pi / (1 - x ** 2) ** .5

    def func(omega):
        omega = np.array(omega) / a
        result = 0
        for i in range(order):
            wt = weight(omega, i)
            ch = chebyshev.chebval(omega, (0,) * i + (1,))
            result += (wt * ch)[np.newaxis, np.newaxis, :] * mu[..., i, np.newaxis]
        return result / a
    func.chebushev_moments = mu

    if return_ncalls:
        return func(-energies.real), matvec_callable.n_calls
    else:
        return func(-energies.real)


def gf_iter(interface, ps, qs, energies, return_ncalls=False, debug=False, **kwargs):
    """Conventional GF iterations."""
    e_vector = list()
    interface.logger.note("Starting a GF calculation n_p x n_q = {:d} x {:d}".format(
        len(ps),
        len(qs),
    ))
    interface.logger.log("Calculating kets ({:d}) ...".format(len(qs)))
    for q in qs:
        e_vector.append(interface.make_e(q))
    matvec_callable = CallCounter(interface.matvec_logging if debug else interface.matvec)
    __ncalls_prev__ = 0
    gfvals = np.zeros((len(ps), len(qs), len(energies)), dtype=complex)
    interface.logger.log("Calculating bras ({:d}) ...".format(len(ps)))
    for ip, p in enumerate(ps):
        b_vector = np.array(interface.make_b(p), dtype=complex)
        for i_energy, energy in enumerate(energies):
            interface.logger.debug1("Calculating bra @ p={:d} e={:d} ... ({:.0f} complete)".format(
                ip, i_energy, 100.0 * (ip*len(energies) + i_energy) / len(ps) / len(energies)))
            size = len(b_vector)
            Ax = spla.LinearOperator((size, size), lambda vector: matvec_callable(vector, energy))
            mx = spla.LinearOperator((size, size), interface.precond(energy))
            x0 = interface.iguess(p, energy)
            if debug:
                interface.reset_log(rhs=b_vector)
            sol, info = spla.gcrotmk(Ax, b_vector, x0=x0, M=mx, **kwargs)
            if debug:
                interface.plot_log()
            interface.logger.debug1("Complete with {:d} matvec calls".format(matvec_callable.n_calls - __ncalls_prev__))
            interface.logger.debug2("Initial guess: {}".format(repr(x0)))
            interface.logger.debug2("Solution: {}".format(repr(sol)))
            interface.logger.debug1("Norm: {:.3e}, initial guess ovlp: {:.3e}, diagonal: {:.3e}".format(
                np.linalg.norm(sol), sol.dot(x0), np.dot(e_vector[ip], sol)))
            __ncalls_prev__ = matvec_callable.n_calls
            if info != 0:
                raise RuntimeError("Error solving linear problem: info={:d}".format(info))
            for iq, q in enumerate(qs):
                gfvals[ip, iq, i_energy] = np.dot(e_vector[iq], sol)
    if return_ncalls:
        return gfvals, matvec_callable.n_calls
    else:
        return gfvals


def greens_func_multiply(ham, vector, linear_part, **kwargs):
    return np.array(ham(vector, **kwargs) + linear_part * vector)


class greens_function:
    def __init__(self, verbose=0):
        self.verbose = verbose

    def td_ip_ao(self, cc, ps, times, mo_coeff, re_im="re", tol=1.e-5):
        """
        E0: total CC gs energy
        ti: initial time
        tf: final time
        times: list of times where GF is computed
        tol : rtol, atol for ODE integrator
                   
        mo_coeff : integrals are assumed in the MO basis
        they are supplied here so we can back transform
        to the AO basis

        re_im in {"re", "im"}

        Signs, etc. Defn. at https://edoc.ub.uni-muenchen.de/18937/1/Wolf_Fabian_A.pdf, pg. 141
        re_im = "re" corresponds to G^<(it) in Eq. A.3, with t = times
        re_im = "im" corresponds to G^<(\tau) in Eq. A.2, with \tau = times        
        """
        dtype = None
        tfac = None
        if re_im == "im":
            dtype = np.float64
            tfac = 1
        elif re_im == "re":
            dtype = np.complex128
            tfac = 1j
        else:
            raise RuntimeError

        # set initial bra/ket
        nmo = mo_coeff.shape[1]
        e_vector_mo = np.zeros([nmo, ip_size(cc)], dtype=dtype)
        for i in range(nmo):
            e_vector_mo[i, :] = greens_e_vector_ip_rhf(cc, i)
        e_vector_ao = np.einsum("pi,ix->px", mo_coeff[ps, :], e_vector_mo)

        b_vector_mo = np.zeros([ip_size(cc), nmo], dtype=dtype)
        for i in range(nmo):
            b_vector_mo[:, i] = greens_b_vector_ip_rhf(cc, i)
        b_vector_ao = np.einsum("xi,ip->xp", b_vector_mo, mo_coeff.T[:, ps])

        # initialize loop variables
        gf_ao = np.zeros((len(ps), len(ps), len(times)), dtype=dtype)
        eomip = pyscf.cc.eom_rccsd.EOMIP(cc)
        eomip_imds = eomip.make_imds()
        ti = times[0]
        tf = times[-1]

        def matr_multiply(t, vector):
            # note: t is a dummy time argument, H is time-independent
            return tfac * np.array(eomip.matvec(vector, imds=eomip_imds))

        for ip, p in enumerate(ps):
            solp = scipy.integrate.solve_ivp(matr_multiply, (ti, tf),
                                             b_vector_ao[:, p], t_eval=times,
                                             rtol=tol, atol=tol)

            for iq, q in enumerate(ps):
                gf_ao[iq, ip, :] = (1.j if re_im == "re" else 1.) * np.dot(e_vector_ao[iq], solp.y)

        return gf_ao

    def td_ea_ao(self, cc, ps, times, mo_coeff, re_im="re", tol=1.e-5):
        """
        See td_ip.
        
        Defn. at https://edoc.ub.uni-muenchen.de/18937/1/Wolf_Fabian_A.pdf, pg. 141
        corresponds to G^>(it) in Eq. A.3
        """
        dtype = None
        tfac = None
        if re_im == "im":
            dtype = np.float64
            tfac = -1
        elif re_im == "re":
            dtype = np.complex128
            tfac = -1j
        else:
            raise RuntimeError

        # set initial bra/ket
        nmo = mo_coeff.shape[1]
        e_vector_mo = np.zeros([nmo, ea_size(cc)], dtype=dtype)
        for i in range(nmo):
            e_vector_mo[i, :] = greens_e_vector_ea_rhf(cc, i)
        e_vector_ao = np.einsum("pi,ix->px", mo_coeff[ps, :], e_vector_mo)

        b_vector_mo = np.zeros([ea_size(cc), nmo], dtype=dtype)
        for i in range(nmo):
            b_vector_mo[:, i] = greens_b_vector_ea_rhf(cc, i)
        b_vector_ao = np.einsum("xi,ip->xp", b_vector_mo, mo_coeff.T[:, ps])

        # initialize loop variables
        gf_ao = np.zeros((len(ps), len(ps), len(times)), dtype=dtype)
        eomea = pyscf.cc.eom_rccsd.EOMEA(cc)
        eomea_imds = eomea.make_imds()
        ti = times[0]
        tf = times[-1]

        def matr_multiply(t, vector):
            # note: t is a dummy time argument, H is time-independent
            return tfac * np.array(eomea.matvec(vector, imds=eomea_imds))

        for ip, p in enumerate(ps):
            solp = scipy.integrate.solve_ivp(matr_multiply, (ti, tf),
                                             b_vector_ao[:, p], t_eval=times,
                                             rtol=tol, atol=tol)

            for iq, q in enumerate(ps):
                gf_ao[iq, ip, :] = (1.j if re_im == "re" else 1.) * np.dot(e_vector_ao[iq], solp.y)

        return gf_ao

    def td_ip(self, cc, ps, qs, times, re_im="re", tol=1.e-5):
        """
        E0: total CC gs energy
        times: list of times where GF is computed
        tol : rtol, atol for ODE integrator
                   
        re_im in {"re", "im"}

        Signs, etc. Defn. at https://edoc.ub.uni-muenchen.de/18937/1/Wolf_Fabian_A.pdf, pg. 141
        re_im = "re" corresponds to G^<(it) in Eq. A.3, with t = times
        re_im = "im" corresponds to G^<(\tau) in Eq. A.2, with \tau = times        
        """
        dtype = None
        tfac = None
        if re_im == "im":
            dtype = np.float64
            tfac = 1
        elif re_im == "re":
            dtype = np.complex128
            tfac = 1j
        else:
            raise RuntimeError

        eomip = pyscf.cc.eom_rccsd.EOMIP(cc)
        eomip_imds = eomip.make_imds()
        ti = times[0]
        tf = times[-1]

        e_vector = list()
        for q in qs:
            e_vector.append(greens_e_vector_ip_rhf(cc, q))

        gfvals = np.zeros((len(ps), len(qs), len(times)), dtype=dtype)

        for ip, p in enumerate(ps):
            b_vector = np.array(greens_b_vector_ip_rhf(cc, p), dtype=dtype)

            def matr_multiply(t, vector, args=None):
                # note: t is a dummy time argument, H is time-independent
                res = tfac * np.array(eomip.matvec(vector, imds=eomip_imds))
                return res

            solp = scipy.integrate.solve_ivp(matr_multiply, (ti, tf),
                                             b_vector, t_eval=times, rtol=tol, atol=tol)

            for iq, q in enumerate(qs):
                gfvals[iq, ip, :] = np.dot(e_vector[iq], solp.y)

        return gfvals

    def td_ea(self, cc, ps, qs, times, re_im="re", tol=1.e-5):
        """
        See td_ip.

        Defn. at https://edoc.ub.uni-muenchen.de/18937/1/Wolf_Fabian_A.pdf, pg. 141
        corresponds to G^>(it) in Eq. A.3
        """
        dtype = None
        tfac = None
        if re_im == "im":
            dtype = np.float64
            tfac = -1
        elif re_im == "re":
            dtype = np.complex128
            tfac = -1j
        else:
            raise RuntimeError

        ti = times[0]
        tf = times[-1]
        eomea = pyscf.cc.eom_rccsd.EOMEA(cc)
        eomea_imds = eomea.make_imds()

        e_vector = list()
        for p in ps:
            e_vector.append(np.array(greens_e_vector_ea_rhf(cc, p), dtype=dtype))
        gfvals = np.zeros((len(ps), len(qs), len(times)), dtype=complex)

        for iq, q in enumerate(qs):
            b_vector = np.array(greens_b_vector_ea_rhf(cc, q), dtype=dtype)

            def matr_multiply(t, vector, args=None):
                # t is a dummy time argument
                res = tfac * np.array(eomea.matvec(vector, imds=eomea_imds))
                return res

            solq = scipy.integrate.solve_ivp(matr_multiply, (ti, tf),
                                             b_vector, t_eval=times, rtol=tol, atol=tol)

            for ip, p in enumerate(ps):
                gfvals[ip, iq, :] = np.dot(e_vector[ip], solq.y)
        return gfvals

    def solve_ip_ao(self, cc, ps, omega_list, mo_coeff, broadening):
        eomip = pyscf.cc.eom_rccsd.EOMIP(cc)
        eomip_imds = eomip.make_imds()
        # GKC: Why is this is the initial guess?
        x0 = initial_ip_guess(cc)
        p0 = 0.0 * x0 + 1.0

        # set initial bra/ket
        nmo = mo_coeff.shape[1]
        e_vector_mo = np.zeros([nmo, ip_size(cc)], dtype=np.complex128)
        for i in range(nmo):
            e_vector_mo[i, :] = greens_e_vector_ip_rhf(cc, i)
        e_vector_ao = np.einsum("pi,ix->px", mo_coeff[ps, :], e_vector_mo)
        b_vector_mo = np.zeros([ip_size(cc), nmo], dtype=np.complex128)
        for i in range(nmo):
            b_vector_mo[:, i] = greens_b_vector_ip_rhf(cc, i)
        b_vector_ao = np.einsum("xi,ip->xp", b_vector_mo, mo_coeff.T[:, ps])
        # initialize loop variables
        gf_ao = np.zeros((len(ps), len(ps), len(omega_list)), dtype=np.complex128)

        for ip, p in enumerate(ps):
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomip.matvec, vector, curr_omega - 1j * broadening, imds=eomip_imds)

                sol = self.__solve_linear_problem__(matr_multiply, b_vector_ao[:, p], x0, p0)
                x0 = sol
                for iq, q in enumerate(ps):
                    gf_ao[ip, iq, iomega] = -np.dot(e_vector_ao[iq, :], sol)
        return gf_ao

    def solve_ea_ao(self, cc, ps, omega_list, mo_coeff, broadening):
        eomea = pyscf.cc.eom_rccsd.EOMEA(cc)
        eomea_imds = eomea.make_imds()
        # GKC: Why is this is the initial guess?
        x0 = initial_ea_guess(cc)
        p0 = 0.0 * x0 + 1.0

        # set initial bra/ket
        nmo = mo_coeff.shape[1]
        e_vector_mo = np.zeros([nmo, ea_size(cc)], dtype=np.complex128)
        for i in range(nmo):
            e_vector_mo[i, :] = greens_e_vector_ea_rhf(cc, i)
        e_vector_ao = np.einsum("pi,ix->px", mo_coeff[ps, :], e_vector_mo)
        b_vector_mo = np.zeros([ea_size(cc), nmo], dtype=np.complex128)
        for i in range(nmo):
            b_vector_mo[:, i] = greens_b_vector_ea_rhf(cc, i)
        b_vector_ao = np.einsum("xi,ip->xp", b_vector_mo, mo_coeff.T[:, ps])
        # initialize loop variables
        gf_ao = np.zeros((len(ps), len(ps), len(omega_list)), dtype=np.complex128)

        for iq, q in enumerate(ps):
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomea.matvec, vector, -curr_omega - 1j * broadening, imds=eomea_imds)

                sol = self.__solve_linear_problem__(matr_multiply, b_vector_ao[:, q], x0, p0)
                x0 = sol
                for ip, p in enumerate(ps):
                    gf_ao[ip, iq, iomega] = np.dot(e_vector_ao[ip], sol)

        return gf_ao

    def solve_ip(self, cc, ps, qs, omega_list, broadening, with_precond=True):
        _s_niter_pep = np.zeros(len(omega_list), dtype=int)
        self.stats = dict(
            niter_per_energy_point=_s_niter_pep,
        )
        eomip = pyscf.cc.eom_rccsd.EOMIP(cc)
        eomip_imds = eomip.make_imds()
        x0 = initial_ip_guess(cc)
        e_vector = list()
        for q in qs:
            e_vector.append(greens_e_vector_ip_rhf(cc, q))
        gfvals = np.zeros((len(ps), len(qs), len(omega_list)), dtype=complex)
        diagonal = ipccsd_diag(eomip, eomip_imds)
        for ip, p in enumerate(ps):
            b_vector = greens_b_vector_ip_rhf(cc, p)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomip.matvec, vector, curr_omega - 1j * broadening, imds=eomip_imds)

                if with_precond:
                    def precond(vector):
                        return vector / (curr_omega + diagonal + 1j * broadening)
                else:
                    precond = None

                matr_multiply = CallCounter(matr_multiply)
                sol = self.__solve_linear_problem__(matr_multiply, b_vector, x0, precond)
                x0 = sol
                _s_niter_pep[iomega] += matr_multiply.n_calls
                for iq, q in enumerate(qs):
                    gfvals[ip, iq, iomega] = -np.dot(e_vector[iq], sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0, 0, :]
        else:
            return gfvals

    def solve_ea(self, cc, ps, qs, omega_list, broadening, with_precond=True):
        _s_niter_pep = np.zeros(len(omega_list), dtype=int)
        self.stats = dict(
            niter_per_energy_point=_s_niter_pep,
        )
        eomea = pyscf.cc.eom_rccsd.EOMEA(cc)
        eomea_imds = eomea.make_imds()
        x0 = initial_ea_guess(cc)
        e_vector = list()
        for p in ps:
            e_vector.append(greens_e_vector_ea_rhf(cc, p))
        gfvals = np.zeros((len(ps), len(qs), len(omega_list)), dtype=complex)
        diagonal = eaccsd_diag(eomea, eomea_imds)
        for iq, q in enumerate(qs):
            b_vector = greens_b_vector_ea_rhf(cc, q)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomea.matvec, vector, -curr_omega - 1j * broadening, imds=eomea_imds)

                if with_precond:
                    def precond(vector):
                        return vector / (- curr_omega + diagonal + 1j * broadening)
                else:
                    precond = None

                matr_multiply = CallCounter(matr_multiply)
                sol = self.__solve_linear_problem__(matr_multiply, b_vector, x0, precond)
                _s_niter_pep[iomega] += matr_multiply.n_calls
                x0 = sol
                for ip, p in enumerate(ps):
                    gfvals[ip, iq, iomega] = np.dot(e_vector[ip], sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0, 0, :]
        else:
            return gfvals

    def solve_gf(self, cc, p, q, omega_list, broadening):
        return self.solve_ip(cc, p, q, omega_list, broadening), self.solve_ea(cc, p, q, omega_list, broadening)

    def __solve_linear_problem__(self, matr_multiply, b_vector, x0, precond=None):
        self.l("  Solving linear problem ...")
        size = len(b_vector)
        Ax = spla.LinearOperator((size, size), matr_multiply)
        if precond is not None:
            mx = spla.LinearOperator((size, size), precond)
        else:
            mx = None
        result, info = spla.gcrotmk(Ax, b_vector, x0=x0, callback=self.__spla_callback__, M=mx, atol=0, tol=1e-1)
        if info != 0:
            raise RuntimeError("Error solving linear problem: info={:d}".format(info))
        return result

    def l(self, x):
        if self.verbose > 0:
            print(x)

    def d(self, x):
        if self.verbose > 1:
            print(x)

    def __spla_callback__(self, r):
        self.d("    res = {}".format(repr(r)))
