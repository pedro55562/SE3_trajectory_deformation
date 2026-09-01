import uaibot as ub

from tree import create_large_tree


def main():

    tree = create_large_tree(
        htm=ub.Utils.trn([0.0, 0.0, 0.0]),
        name="large_tree",
        height=8.0,
        crown_radius=3.0,
        trunk_radius=0.45,
        branches_per_level=5,
        seed=7,
    )

    sim = ub.Simulation.create_sim_grid(tree)

    sim.run()


if __name__ == "__main__":
    main()