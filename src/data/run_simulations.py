import argparse
from utils import run_n_simulations, create_gif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_particles", default=10, type=int)
    parser.add_argument("-n_sim", default=1, type=int)
    args = parser.parse_args()

    sim_path = f"data/raw/nbody_{args.n_particles}_particles"
    run_n_simulations(
        output_path=sim_path, n_particles=args.n_particles, n_sim=args.n_sim
    )
    create_gif(sim_path+"_sim0000.npy", "gifs/test2.gif", n_particles=args.n_particles, trunc=False)
