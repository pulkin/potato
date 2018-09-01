import unittest
import numpy
from matplotlib import pyplot, gridspec

from pyscf import gto, scf, cc
from greens_function import gf_retarded_convention, EOMIPGF, EOMEAGF, gf_chebushev, gf_fourier, gf_iter


class H2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.verbose = 6
        cls.mol.atom = "H 0 0 0; H 0.74 0 0"
        cls.mol.basis = 'ccpvdz'
        cls.mol.build()
        cls.mol.max_memory = 16e3

        cls.mf = scf.RHF(cls.mol)
        cls.mf.kernel()

        cls.ccsd = cc.RCCSD(cls.mf)
        cls.ccsd.kernel()
        cls.ccsd.solve_lambda()
        cls.eip, _ = cls.ccsd.ipccsd(nroots=4)
        cls.eea, _ = cls.ccsd.eaccsd(nroots=6)

        cls.energies = numpy.linspace(-1.9, 1, 200)
        cls.eta = 0.01
        cls.title = "H2"

        cls.options = dict(
            iter=dict(),
            chebushev=dict(order=400, range=10),
            td=dict(factor=1),
        )

    def test_(self):
        import warnings
        warnings.filterwarnings("error")
        self._test_()

    def _test_(self):
        basis = numpy.arange(self.ccsd.nmo)
        energies = gf_retarded_convention(self.energies, self.eta)

        if_ip = EOMIPGF(self.ccsd)
        if_ea = EOMEAGF(self.ccsd)

        print "Number of calls:"

        if "iter" in self.options:

            options = self.options["iter"]
            gfip, gfip_ncalls = gf_iter(if_ip, basis, basis, energies, return_ncalls=True, atol=options.get("atol", 1e-5))
            # niter_ip = gf.stats["niter_per_energy_point"]
            # gfip_ncalls = sum(niter_ip)
            gfea, gfea_ncalls = gf_iter(if_ea, basis, basis, energies, return_ncalls=True, atol=options.get("atol", 1e-5))
            # niter_ea = gf.stats["niter_per_energy_point"]
            # gfea_ncalls = sum(niter_ea)
            dip = -numpy.trace(gfip).imag
            dea = -numpy.trace(gfea).imag

            total_calls = gfip_ncalls + gfea_ncalls
            total_points = len(energies) * len(basis)
            print " - sc: {:d}, {:.1f} per energy/basis point ({:.1f} IP, {:.1f} EA)".format(
                total_calls, 1.0 * total_calls / total_points, 1.0 * gfip_ncalls / total_points, 1.0 * gfea_ncalls / total_points)

        if "chebushev" in self.options:

            options = self.options["chebushev"]
            gfipch, gfip_ncalls = gf_chebushev(if_ip, basis, basis, energies, options.get("order", 100), options.get("range", 10), return_ncalls=True)
            gfeach, gfea_ncalls = gf_chebushev(if_ea, basis, basis, energies, options.get("order", 100), options.get("range", 10), return_ncalls=True)
            dipch = -numpy.trace(gfipch)
            deach = -numpy.trace(gfeach)

            total_calls = gfip_ncalls + gfea_ncalls
            total_points = len(energies) * len(basis)
            print " - ch: {:d}, {:.1f} per energy/basis point ({:.1f} IP, {:.1f} EA)".format(
                total_calls, 1.0 * total_calls / total_points, 1.0 * gfip_ncalls / total_points, 1.0 * gfea_ncalls / total_points)

        if "td" in self.options:

            options = self.options["td"]
            gfiptd, gfip_ncalls = gf_fourier(if_ip, basis, basis, energies, sampling_factor=options.get("factor", 1), return_ncalls=True)
            gfeatd, gfea_ncalls = gf_fourier(if_ea, basis, basis, energies, sampling_factor=options.get("factor", 1), return_ncalls=True)
            diptd = -numpy.trace(gfiptd).imag
            deatd = -numpy.trace(gfeatd).imag

            total_calls = gfip_ncalls + gfea_ncalls
            total_points = len(energies) * len(basis)
            print " - td: {:d}, {:.1f} per energy/basis point ({:.1f} IP, {:.1f} EA)".format(
                total_calls, 1.0 * total_calls / total_points, 1.0 * gfip_ncalls / total_points, 1.0 * gfea_ncalls / total_points)

        pyplot.figure(dpi=150)
        gs = gridspec.GridSpec(3, 1, height_ratios=(2, 2, 1))
        gs.update(hspace=0)
        ax = pyplot.subplot(gs[0])
        ax.set_title(self.title)
        for i in self.eip:
            ax.axvline(x=-i, ls="--")
        if "iter" in self.options:
            ax.plot(energies.real, dip, color="black", label="sc")
        if "chebushev" in self.options:
            ax.plot(energies.real, dipch, ls="--", color="#5555FF", label="Chebushev", lw=2)
        if "td" in self.options:
            ax.plot(energies.real, diptd, ls=":", color="#5555FF", label="Fourier", lw=2)
        ax.set_ylabel("Intensity (arb. units)")
        ax.text(0.9, 0.4, "(" + str(gfip_ncalls) + ")", transform=ax.transAxes, backgroundcolor="white")
        pyplot.legend()

        ax = pyplot.subplot(gs[1])
        for i in self.eea:
            ax.axvline(x=i, ls="--")
        if "iter" in self.options:
            ax.plot(energies.real, dea, color="black")
        if "chebushev" in self.options:
            ax.plot(energies.real, deach, ls="--", color="#FF5555", lw=2)
        if "td" in self.options:
            ax.plot(energies.real, deatd, ls=":", color="#FF5555", lw=2)
        ax.text(0.9, 0.4, "(" + str(gfea_ncalls) + ")", transform=ax.transAxes, backgroundcolor="white")
        ax.set_ylabel("Intensity (arb. units)")

        if "iter" in self.options:
            ax = pyplot.subplot(gs[2], sharex=pyplot.gca())
            # ax.semilogy(energies, niter_ip, color="#5555FF")
            # ax.semilogy(energies, niter_ea, color="#FF5555")
            pyplot.xlabel("Energy (Hartree)")
            pyplot.ylabel("iter")
            pyplot.ylim(bottom=0)

        pyplot.tight_layout()
        pyplot.savefig("{}.pdf".format(self.title))
        pyplot.savefig("{}.png".format(self.title))
        pyplot.show()


class H2OTests(H2Tests):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.verbose = 4
        cls.mol.atom = "O 0 0 0; H 0.790689766 0 0.612217330; H -0.790689766 0 0.612217330"
        cls.mol.basis = '3-21g'
        cls.mol.build()
        cls.mol.max_memory = 16e3
        cls.mol.incore_anyway = True
        cls.mol.unit = "angstrom"

        cls.mf = scf.RHF(cls.mol)
        cls.mf.kernel()

        cls.ccsd = cc.RCCSD(cls.mf, frozen=1)
        cls.ccsd.kernel()
        cls.ccsd.solve_lambda()
        cls.eip, _ = cls.ccsd.ipccsd(nroots=8)
        cls.eea, _ = cls.ccsd.eaccsd(nroots=4)

        cls.energies = numpy.linspace(-1.9, 1, 50)
        cls.eta = 0.04
        cls.title = 'H2O ({})'.format(cls.mol.basis)
        cls.options = dict(
            iter=dict(),
            chebushev=dict(order=400, range=10),
            td=dict(factor=1),
        )
