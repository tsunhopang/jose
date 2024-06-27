"""
Provide an easy example of the TOV solver and provide a timing measurement.
"""
# The following import is needed on LDAS@CIT, remove if breaking
import psutil
p = psutil.Process()
p.cpu_affinity([0])

# Regular imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1 to force CPU
import jax
import numpy as np
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print(f"Devices found found? {jax.devices()}")

from joseTOV import eos

def test_tov():
    working_dir = os.path.dirname(__file__)
    data_dir = os.path.join(working_dir, "data")

    # Load the eos files
    micro_eos = jnp.load(
        f"{data_dir}/micro_eos.npz",
    )
    # create the eos object
    eos_input = eos.Interpolate_EOS_model(
        micro_eos["n"], micro_eos["p"], micro_eos["e"]
    )
    eos_tuple = (
        eos_input.n,
        eos_input.p,
        eos_input.h,
        eos_input.e,
        eos_input.dloge_dlogp,
    )
    # solve the TOV equation for M-R-Lambda curve
    print("Constructing the family")
    logpcs, ms, rs, Lambdas = eos.construct_family(eos_tuple)

    print(np.shape(logpcs))
    print(np.shape(ms))
    print(np.shape(rs))
    print(np.shape(Lambdas))
    
    if not os.path.exists("./figures/"):
        os.makedirs("./figures/")
        
    # Make some plots
    plt.figure()
    plt.plot(rs, ms)
    plt.xlabel("Radius [km]")
    plt.ylabel("Mass [Msun]")
    plt.savefig("./figures/mass_radius.png", dpi=150)
    plt.close()
    
    plt.figure()
    plt.plot(rs, Lambdas)
    plt.xlabel("Radius [km]")
    plt.ylabel("Lambda")
    plt.savefig("./figures/radius_lambda.png", dpi=150)
    plt.close()
    
    plt.figure()
    plt.plot(ms, Lambdas)
    plt.xlabel("Mass [Msun]")
    plt.ylabel("Lambda")
    plt.savefig("./figures/mass_lambda.png", dpi=150)
    plt.close()
    

def main():
    test_tov()
    
if __name__ == "__main__":
    main()