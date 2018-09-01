import unittest
import numpy
from numpy import testing
from matplotlib import pyplot
import os

from pyscf.pbc import gto, scf, cc
from pyscf.pbc.tools.pbc import super_cell

import k_greens_function as kgf
import greens_function as gf


class SupercellTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        k_grid = [2, 1, 1]

        # Unit cell
        cls.cell = gto.Cell()
        cls.cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
        cls.cell.basis = 'gth-szv'
        cls.cell.pseudo = 'gth-pade'
        cls.cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000
        '''
        cls.cell.unit = 'B'
        cls.cell.verbose = 7
        cls.cell.mesh = [24] * 3
        cls.cell.build()

        # MF
        cls.mf = scf.KRHF(cls.cell, kpts=cls.cell.make_kpts(k_grid), exxdiv=None)
        cls.mf.chkfile = 'test_kgf_proper_rhf_chk.dat'
        if os.path.exists(cls.mf.chkfile):
            cls.mf.update()
        else:
            cls.mf.kernel()

        # Supercell
        cls.cell2 = super_cell(cls.cell, k_grid)
        cls.mf2 = scf.KRHF(cls.cell2, exxdiv=None)
        cls.order = numpy.argsort(numpy.hstack(cls.mf.mo_energy))
        c1 = cls.mf.mo_coeff[0]
        c2 = cls.mf.mo_coeff[1]
        c1, c2 = numpy.concatenate((c1, c1), axis=0) / 2.**.5, numpy.concatenate((c2, -c2), axis=0) / 2.**.5
        cls.mf2.mo_coeff = [numpy.concatenate((c1, c2), axis=1)[:, cls.order]]
        cls.mf2.mo_energy = [numpy.concatenate(cls.mf.mo_energy)[cls.order]]
        cls.mf2.mo_occ = [numpy.concatenate(cls.mf.mo_occ)[cls.order]]
        cls.mf2.make_rdm1()

        # CCSD
        cls.ccsd = cc.KRCCSD(cls.mf)
        cls.ccsd.conv_tol_normt = 1e-8
        cls.ccsd.kernel()

        cls.ccsd2 = cc.KRCCSD(cls.mf2)
        cls.ccsd2.conv_tol_normt = 1e-8
        cls.ccsd2.kernel()

        # TODO: lambda iterations
        cls.ccsd.l1 = cls.ccsd.t1
        cls.ccsd.l2 = cls.ccsd.t2
        # cls.ccsd.ipccsd(nroots=2)
        # cls.ccsd.eaccsd(nroots=2)
        # cls.ip_samples = [-0.71, -1]
        cls.ip_samples = [-0.71]
        cls.ea_samples = [1.13, 0.3]
        cls.eta = 0.01

        # =====
        cls.nocc, cls.nvirt = cls.ccsd.t1[0].shape
        cls.nmo = cls.nocc + cls.nvirt
        cls.o1 = cls.order[:2*cls.nocc] < cls.nmo
        cls.o2 = cls.order[:2*cls.nocc] >= cls.nmo
        cls.v1 = cls.order[2*cls.nocc:] < cls.nmo
        cls.v2 = cls.order[2*cls.nocc:] >= cls.nmo
        cls.a1 = cls.order < cls.nmo
        cls.a2 = cls.order >= cls.nmo
        # =====

        cls.ccsd2.l1 = cls.ccsd2.t1
        cls.ccsd2.l2 = cls.ccsd2.t2

    def test_t1(self):
        a11, a12 = self.ccsd.t1
        a2, = self.ccsd2.t1

        testing.assert_allclose(a2[numpy.ix_(self.o1, self.v1)], a11, atol=1e-8)
        testing.assert_allclose(a2[numpy.ix_(self.o2, self.v2)], a12, atol=1e-8)
        testing.assert_allclose(a2[numpy.ix_(self.o1, self.v2)], 0, atol=1e-13)
        testing.assert_allclose(a2[numpy.ix_(self.o2, self.v1)], 0, atol=1e-13)

    def test_t2(self):
        (((a1_1111, a1_1122), (a1_1212, a1_1221)), ((a1_2112, a1_2121), (a1_2211, a1_2222))) = self.ccsd.t2
        (((a2,),),) = self.ccsd2.t2

        o1, o2, v1, v2 = self.o1, self.o2, self.v1, self.v2

        testing.assert_allclose(a2[numpy.ix_(o1, o1, v1, v1)], a1_1111, atol=1e-8)
        testing.assert_allclose(a2[numpy.ix_(o1, o1, v2, v2)], a1_1122, atol=1e-8)

        testing.assert_allclose(a2[numpy.ix_(o1, o2, v1, v2)], a1_1212, atol=1e-8)
        testing.assert_allclose(a2[numpy.ix_(o1, o2, v2, v1)], a1_1221, atol=1e-8)

        testing.assert_allclose(a2[numpy.ix_(o2, o1, v1, v2)], a1_2112, atol=1e-8)
        testing.assert_allclose(a2[numpy.ix_(o2, o1, v2, v1)], a1_2121, atol=1e-8)

        testing.assert_allclose(a2[numpy.ix_(o2, o2, v1, v1)], a1_2211, atol=1e-8)
        testing.assert_allclose(a2[numpy.ix_(o2, o2, v2, v2)], a1_2222, atol=1e-8)

        testing.assert_allclose(a2[numpy.ix_(o1, o1, v1, v2)], 0, atol=1e-13)
        testing.assert_allclose(a2[numpy.ix_(o1, o1, v2, v1)], 0, atol=1e-13)

        testing.assert_allclose(a2[numpy.ix_(o1, o2, v1, v1)], 0, atol=1e-13)
        testing.assert_allclose(a2[numpy.ix_(o1, o2, v2, v2)], 0, atol=1e-13)

        testing.assert_allclose(a2[numpy.ix_(o2, o1, v1, v1)], 0, atol=1e-13)
        testing.assert_allclose(a2[numpy.ix_(o2, o1, v2, v2)], 0, atol=1e-13)

        testing.assert_allclose(a2[numpy.ix_(o2, o2, v1, v2)], 0, atol=1e-13)
        testing.assert_allclose(a2[numpy.ix_(o2, o2, v2, v1)], 0, atol=1e-13)

    def test_vectors(self, atol=1e-8, zerotol=1e-13):

        o1, o2 = self.o1, self.o2
        v1, v2 = self.v1, self.v2

        for m1, m2, v2a, bvec, name in (
                (o1, o2, self.ccsd.__class__.ip_vector_to_amplitudes, kgf.greens_b_vector_ip_krhf, "IP b"),
                (o1, o2, self.ccsd.__class__.ip_vector_to_amplitudes, kgf.greens_e_vector_ip_krhf, "IP e"),
                (v1, v2, self.ccsd.__class__.ea_vector_to_amplitudes, kgf.greens_b_vector_ea_krhf, "EA b"),
                (v1, v2, self.ccsd.__class__.ea_vector_to_amplitudes, kgf.greens_e_vector_ea_krhf, "EA e"),
        ):
            print "Testing {} ...".format(name)

            for i in range(2 * self.nmo):

                k = self.order[i] // self.nmo
                i_k = self.order[i] % self.nmo

                if k == 0:

                    big1, ((big2,),) = v2a(self.ccsd2, bvec(self.ccsd2, i, 0))
                    small_00, ((small_0000, small_0110), (small_1010, small_1100)) = v2a(self.ccsd, bvec(self.ccsd, i_k, k))

                    testing.assert_allclose(big1[m1], small_00, atol=atol)  # K = 0-0
                    testing.assert_allclose(big1[m2], 0, atol=zerotol)      # K = 1-0

                    # These do not conserve k (the last index is always v1)
                    testing.assert_allclose(big2[numpy.ix_(o1, m1, v2)], 0, atol=zerotol)  # K = 001-0
                    testing.assert_allclose(big2[numpy.ix_(o1, m2, v1)], 0, atol=zerotol)  # K = 010-0
                    testing.assert_allclose(big2[numpy.ix_(o2, m1, v1)], 0, atol=zerotol)  # K = 100-0
                    testing.assert_allclose(big2[numpy.ix_(o2, m2, v2)], 0, atol=zerotol)  # K = 111-0

                    # These conserve k
                    testing.assert_allclose(big2[numpy.ix_(o1, m1, v1)], small_0000, atol=atol)  # K = 000-0
                    testing.assert_allclose(big2[numpy.ix_(o1, m2, v2)], small_0110, atol=atol)  # K = 011-0
                    testing.assert_allclose(big2[numpy.ix_(o2, m1, v2)], small_1010, atol=atol)  # K = 101-0
                    testing.assert_allclose(big2[numpy.ix_(o2, m2, v1)], small_1100, atol=atol)  # K = 110-0

                if k == 1:

                    big1, ((big2,),) = v2a(self.ccsd2, bvec(self.ccsd2, i, 0))
                    small_11, ((small_0011, small_0101), (small_1001, small_1111)) = v2a(self.ccsd, bvec(self.ccsd, i_k, k))

                    testing.assert_allclose(big1[m1], 0, atol=zerotol)      # K = 0-1
                    testing.assert_allclose(big1[m2], small_11, atol=atol)  # K = 1-1

                    # These do not conserve k (the last index is always v1)
                    testing.assert_allclose(big2[numpy.ix_(o1, m1, v1)], 0, atol=zerotol)  # K = 000-1
                    testing.assert_allclose(big2[numpy.ix_(o1, m2, v2)], 0, atol=zerotol)  # K = 011-1
                    testing.assert_allclose(big2[numpy.ix_(o2, m1, v2)], 0, atol=zerotol)  # K = 101-1
                    testing.assert_allclose(big2[numpy.ix_(o2, m2, v1)], 0, atol=zerotol)  # K = 110-1

                    # These conserve k
                    testing.assert_allclose(big2[numpy.ix_(o1, m1, v2)], small_0011, atol=atol)  # K = 001-1
                    testing.assert_allclose(big2[numpy.ix_(o1, m2, v1)], small_0101, atol=atol)  # K = 010-1
                    testing.assert_allclose(big2[numpy.ix_(o2, m1, v1)], small_1001, atol=atol)  # K = 100-1
                    testing.assert_allclose(big2[numpy.ix_(o2, m2, v2)], small_1111, atol=atol)  # K = 111-1

    def test_ip_gf(self, atol=2e-4, zerotol=2e-4):
        basis = numpy.arange(self.nmo)
        basis2 = numpy.arange(2 * self.nmo)
        # basis = [1]

        small = kgf.gf_iter(
            kgf.k_interfaces(kgf.EOMIPGFatK, self.ccsd, [0, 1]),
            basis, basis, gf.gf_retarded_convention(self.ip_samples, self.eta),
        )
        big = kgf.gf_iter(
            kgf.k_interfaces(kgf.EOMIPGFatK, self.ccsd2, [0]),
            basis2, basis2, gf.gf_retarded_convention(self.ip_samples, self.eta),
        )

        for i_e in range(len(self.ip_samples)):
            s = small[..., i_e]
            b = big[0, ..., i_e]
            testing.assert_allclose(b[numpy.ix_(self.a1, self.a1)], s[0], atol=atol)
            testing.assert_allclose(b[numpy.ix_(self.a2, self.a2)], s[1], atol=atol)
            testing.assert_allclose(b[numpy.ix_(self.a1, self.a2)], 0, atol=zerotol)
            testing.assert_allclose(b[numpy.ix_(self.a2, self.a1)], 0, atol=zerotol)


class DiamondTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = gto.Cell()
        cls.cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
        cls.cell.basis = 'gth-szv'
        cls.cell.pseudo = 'gth-pade'
        cls.cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cls.cell.unit = 'B'
        cls.cell.build()
        cls.cell.verbose = 0
        cls.cell.incore_anyway = True
        cls.cell.max_memory = 20e3

        cls.mf = scf.KRHF(cls.cell, kpts=cls.cell.make_kpts([2, 1, 1]))
        cls.mf.chkfile = 'test_kgf_proper_rhf_chk.dat'
        if os.path.exists(cls.mf.chkfile):
            cls.mf.update()
        else:
            cls.mf.kernel()

        cls.ccsd = cc.KRCCSD(cls.mf, frozen=(cls.cell.nelectron - 4) // 2)
        cls.ccsd.kernel()
        # TODO: lambda iterations
        cls.ccsd.l1 = cls.ccsd.t1
        cls.ccsd.l2 = cls.ccsd.t2
        cls.eip, cls.ip_vecs = cls.ccsd.ipccsd(nroots=2)
        cls.eea, cls.ea_vecs = cls.ccsd.eaccsd(nroots=2)

        cls.energies = numpy.linspace(0, 1.3, 50)
        cls.eta = 0.04
        cls.title = "diamond"
        cls.options = ("iter", "td")

    def test_(self):
        kpts = numpy.arange(self.ccsd.nkpts)
        basis = numpy.arange(sum(self.ccsd.t1[0].shape))
        energies = gf.gf_retarded_convention(self.energies, self.eta)

        print "Number of calls:"

        if "iter" in self.options:

            gfip, gfip_ncalls = kgf.gf_iter(
                kgf.k_interfaces(kgf.EOMIPGFatK, self.ccsd, kpts),
                basis, basis, energies, return_ncalls=True)
            gfea, gfea_ncalls = kgf.gf_iter(
                kgf.k_interfaces(kgf.EOMEAGFatK, self.ccsd, kpts),
                basis, basis, energies, return_ncalls=True)
            dip = - numpy.trace(gfip, axis1=1, axis2=2).imag
            dea = - numpy.trace(gfea, axis1=1, axis2=2).imag

            total_calls = gfip_ncalls + gfea_ncalls
            total_points = len(energies) * len(basis) * len(kpts)
            print " - sc: {:d}, {:.1f} per energy/basis point ({:.1f} IP, {:.1f} EA)".format(
                total_calls, 1.0 * total_calls / total_points, 1.0 * gfip_ncalls / total_points, 1.0 * gfea_ncalls / total_points)

        if "td" in self.options:

            gfiptd, gfiptd_ncalls = kgf.gf_fourier(
                kgf.k_interfaces(kgf.EOMIPGFatK, self.ccsd, kpts),
                basis, basis, energies, return_ncalls=True)
            gfeatd, gfeatd_ncalls = kgf.gf_fourier(
                kgf.k_interfaces(kgf.EOMEAGFatK, self.ccsd, kpts),
                basis, basis, energies, return_ncalls=True)
            diptd = - numpy.trace(gfiptd, axis1=1, axis2=2).imag
            deatd = - numpy.trace(gfeatd, axis1=1, axis2=2).imag

            total_calls = gfiptd_ncalls + gfeatd_ncalls
            total_points = len(energies) * len(basis) * len(kpts)
            print " - ft: {:d}, {:.1f} per energy/basis point ({:.1f} IP, {:.1f} EA)".format(
                total_calls, 1.0 * total_calls / total_points, 1.0 * gfiptd_ncalls / total_points, 1.0 * gfeatd_ncalls / total_points)

        fig, axes = pyplot.subplots(self.ccsd.nkpts, sharex=True, sharey=True, gridspec_kw=dict(
            hspace=0,
        ))
        if self.ccsd.nkpts == 1:
            axes = [axes]
        for idx, (eip_k, eea_k, emf_k, omf_k, ax) in enumerate(zip(self.eip, self.eea, self.mf.mo_energy, self.mf.mo_occ, axes)):
            for i in eip_k:
                ax.axvline(x=-i, ls="--", color="#5555FF")
            for i in eea_k:
                ax.axvline(x=i, ls="--", color="#FF5555")
            for i, j in zip(emf_k, omf_k):
                ax.axvline(x=i, color="gray")
            if "iter" in self.options:
                ax.plot(self.energies, dip[idx], color="#5555FF", label="sc")
                ax.plot(self.energies, dea[idx], color="#FF5555")

            if "td" in self.options:
                ax.plot(self.energies, diptd[idx], color="#5555FF", ls=":", lw=2, label="fourier")
                ax.plot(self.energies, deatd[idx], color="#FF5555", ls=":", lw=2)
            ax.set_ylabel("Intensity (arb. units)")
            if ax is not axes[-1]:
                pyplot.setp(ax.get_xticklabels(), visible=False)
            ax.text(0.9, 0.4, "k #{:d}".format(idx), transform=ax.transAxes, backgroundcolor="white")
        axes[-1].set_xlabel("Energy (Hartree)")
        axes[0].set_title(self.title)
        axes[0].legend()
        pyplot.tight_layout()
        pyplot.savefig("{}.pdf".format(self.title))
        pyplot.savefig("{}.png".format(self.title))
        pyplot.show()


class AlTest(DiamondTests):
    @classmethod
    def setUpClass(cls):
        cls.cell = gto.Cell()
        cls.cell.atom = '''Al 0 0 0'''
        cls.cell.basis = 'gth-szv'
        cls.cell.pseudo = 'gth-pade'
        cls.cell.a = '''
        2.47332919 0 1.42797728
        0.82444306 2.33187713 1.42797728
        0, 0, 2.85595455
        '''
        cls.cell.unit = 'angstrom'
        cls.cell.build()
        cls.cell.verbose = 4
        cls.cell.incore_anyway = True
        cls.cell.max_memory = 20e3

        cls.mf = scf.KRHF(cls.cell, kpts=cls.cell.make_kpts([2, 1, 1], scaled_center=(.1, .2, .3)))
        cls.mf.chkfile = 'test_kgf_proper_rhf_al_chk.dat'
        if os.path.exists(cls.mf.chkfile):
            cls.mf.update()
        else:
            cls.mf.kernel()

        cls.ccsd = cc.KRCCSD(cls.mf)
        cls.ccsd.kernel()
        # TODO: lambda iterations
        cls.ccsd.l1 = cls.ccsd.t1
        cls.ccsd.l2 = cls.ccsd.t2
        cls.eip, cls.ip_vecs = cls.ccsd.ipccsd(nroots=2)
        cls.eea, cls.ea_vecs = cls.ccsd.eaccsd(nroots=2)

        cls.energies = numpy.linspace(-1, 1.3, 100)
        cls.eta = 0.04
        cls.title = "Al"
        cls.options = ("iter", "td")


class LargeTest(DiamondTests):
    @classmethod
    def setUpClass(cls):
        cls.cell = gto.Cell()
        cls.cell.atom = '''Al 0 0 0'''
        cls.cell.basis = 'gth-szv'
        cls.cell.pseudo = 'gth-pade'
        cls.cell.a = '''
        2.47332919 0 1.42797728
        0.82444306 2.33187713 1.42797728
        0 0 2.85595455
        '''
        cls.cell.unit = 'angstrom'
        cls.cell.build()
        cls.cell.verbose = 4
        cls.cell.incore_anyway = True
        cls.cell.max_memory = 20e3

        cls.mf = scf.KRHF(cls.cell, kpts=cls.cell.make_kpts([2, 2, 2], scaled_center=(.1, .2, .3)))
        cls.mf.chkfile = 'test_kgf_proper_rhf_al2_chk.dat'
        if os.path.exists(cls.mf.chkfile):
            cls.mf.update()
        else:
            cls.mf.kernel()

        cls.ccsd = cc.KRCCSD(cls.mf)
        cls.ccsd.kernel()
        # TODO: lambda iterations
        cls.ccsd.l1 = cls.ccsd.t1
        cls.ccsd.l2 = cls.ccsd.t2
        cls.eip, cls.ip_vecs = cls.ccsd.ipccsd(nroots=2)
        cls.eea, cls.ea_vecs = cls.ccsd.eaccsd(nroots=2)

        cls.energies = numpy.linspace(-1, 1.3, 100)
        cls.eta = 0.04
        cls.title = "Al-2"
        cls.options = ("td",)
