
# FIXME: this script is outdated and does not work anymore with the current API
from . import eos


def MetaModel_with_CSE_parameters_to_family(
    coefficient_sat,
    coefficient_sym,
    n_break,
    ngrids,
    cs2grids,
    nsat,
    nmax,
):

    eos_model = eos.MetaModel_with_CSE_EOS_model(
        coefficient_sat,
        coefficient_sym,
        n_break,
        ngrids,
        cs2grids,
        nsat=nsat,
        nmax=nmax,
    )

    eos_tuple = (
        eos_model.n,
        eos_model.p,
        eos_model.h,
        eos_model.e,
        eos_model.dloge_dlogp,
    )
    logpcs, ms, rs, Lambdas = eos.construct_family(eos_tuple)

    return logpcs, ms, rs, Lambdas
