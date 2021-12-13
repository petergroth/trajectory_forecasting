import gif
import matplotlib.pyplot as plt
import nbody as nb
import numpy as np
import tensorflow as tf
from matplotlib.patches import Circle


def generate_fully_connected_edges(n_nodes: int):
    # Generates edge indices for a fully connected graph. Includes self-loops. All edges are bi-directional
    edge_index = np.array(np.meshgrid(np.arange(n_nodes), np.arange(n_nodes))).reshape(
        2, -1
    )
    return edge_index


def parse_sequence(data):
    # Example field definition
    roadgraph_features = {
        "roadgraph_samples/dir": tf.io.FixedLenFeature(
            [20000, 3], tf.float32, default_value=None
        ),
        "roadgraph_samples/id": tf.io.FixedLenFeature(
            [20000, 1], tf.int64, default_value=None
        ),
        "roadgraph_samples/type": tf.io.FixedLenFeature(
            [20000, 1], tf.int64, default_value=None
        ),
        "roadgraph_samples/valid": tf.io.FixedLenFeature(
            [20000, 1], tf.int64, default_value=None
        ),
        "roadgraph_samples/xyz": tf.io.FixedLenFeature(
            [20000, 3], tf.float32, default_value=None
        ),
    }

    # Features of other agents.
    state_features = {
        "state/id": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        "state/type": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        "state/is_sdc": tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        "state/tracks_to_predict": tf.io.FixedLenFeature(
            [128], tf.int64, default_value=None
        ),
        "state/current/bbox_yaw": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/height": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/length": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/timestamp_micros": tf.io.FixedLenFeature(
            [128, 1], tf.int64, default_value=None
        ),
        "state/current/valid": tf.io.FixedLenFeature(
            [128, 1], tf.int64, default_value=None
        ),
        "state/current/vel_yaw": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/velocity_x": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/velocity_y": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/width": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/x": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/y": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/z": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/future/bbox_yaw": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/height": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/length": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/timestamp_micros": tf.io.FixedLenFeature(
            [128, 80], tf.int64, default_value=None
        ),
        "state/future/valid": tf.io.FixedLenFeature(
            [128, 80], tf.int64, default_value=None
        ),
        "state/future/vel_yaw": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/velocity_x": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/velocity_y": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/width": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/x": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/y": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/z": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/past/bbox_yaw": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/height": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/length": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/timestamp_micros": tf.io.FixedLenFeature(
            [128, 10], tf.int64, default_value=None
        ),
        "state/past/valid": tf.io.FixedLenFeature(
            [128, 10], tf.int64, default_value=None
        ),
        "state/past/vel_yaw": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/velocity_x": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/velocity_y": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/width": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/x": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/y": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/z": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
    }

    traffic_light_features = {
        "traffic_light_state/current/state": tf.io.FixedLenFeature(
            [1, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/current/valid": tf.io.FixedLenFeature(
            [1, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/current/x": tf.io.FixedLenFeature(
            [1, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/current/y": tf.io.FixedLenFeature(
            [1, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/current/z": tf.io.FixedLenFeature(
            [1, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/past/state": tf.io.FixedLenFeature(
            [10, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/past/valid": tf.io.FixedLenFeature(
            [10, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/past/x": tf.io.FixedLenFeature(
            [10, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/past/y": tf.io.FixedLenFeature(
            [10, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/past/z": tf.io.FixedLenFeature(
            [10, 16], tf.float32, default_value=None
        ),
    }

    features_description = {}
    features_description.update(roadgraph_features)
    features_description.update(state_features)
    features_description.update(traffic_light_features)

    return tf.io.parse_single_example(data, features_description)


def run_simulation(T=30, dt=0.5):
    n_particles = np.random.random_integers(6, 10)

    # Define parameters for simulation
    n_dim = 2
    pos_lim = [-80, 80]  # min/max values in x/y
    v_scale = 15  # Scale of velocity
    m_loc = 3e10  # Mass mean
    m_scale = 5e9  # Mass scale

    # Initial conditions
    x0 = np.random.uniform(pos_lim[0], pos_lim[1], (n_particles, n_dim))
    x0 -= x0.mean(0)
    v0 = np.random.normal(loc=0, scale=v_scale, size=(n_particles, n_dim))
    v0 = v0 - np.mean(v0)  # Recenter
    w0 = np.zeros((n_particles, 1))
    m = np.abs(np.random.normal(m_loc, m_scale, (n_particles, 1)))
    q = np.random.normal(0, 1e-3, (n_particles, 1))
    # q = np.zeros((n_particles, 1))
    r = m / m_loc * 2

    # Initialise system
    nbody_system = nb.spheres(x0, v0, w0, m, q, r)

    # Solving the system
    nbody_system.solve(T, dt, collision=True, debug=False)

    # Extract solutions
    positions = nbody_system.x
    velocities = nbody_system.v
    sizes = nbody_system.r

    return positions, velocities, sizes


def run_n_simulations(output_path, n_particles=10, n_sim=1, n_steps=91):
    freq = 0.1
    T = int(np.ceil(freq * n_steps))
    dt = 0.1

    for idx in range(n_sim):
        positions, velocities, sizes = run_simulation(T=T, dt=dt)
        # Allocate array
        full_arr = np.zeros((n_steps, positions.shape[1], 5))
        full_arr[:, :, :2] = positions[:n_steps]
        full_arr[:, :, 2:4] = velocities[:n_steps]
        full_arr[:, :, 4] = sizes.squeeze()

        if output_path.endswith(".npy"):
            output_path = output_path[:-4]
        np.save(f"{output_path}_sim{idx:04}.npy", full_arr)


@gif.frame
def create_frame(t, n_particles, positions, velocities, sizes):
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(n_particles):
        ax.add_patch(
            Circle(
                positions[t, i, :], sizes[0, i], edgecolor="k", facecolor="k", alpha=0.5
            )
        )
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.quiver(
        positions[t, :, 0],
        positions[t, :, 1],
        velocities[t, :, 0],
        velocities[t, :, 1],
        width=0.003,
        headwidth=5,
        angles="xy",
        scale_units="xy",
        scale=1.0,
    )
    ax.set(xlim=(-100, 100), ylim=(-100, 100))
    ax.set_title(f"Positions and velocities at time {t=}")


def create_gif(input_path, output_path, n_particles, trunc=False):
    full_arr = load_simulations(path=input_path)
    positions = full_arr[:, :, :2]
    velocities = full_arr[:, :, 2:4]
    sizes = full_arr[:, :, 4]

    # Animate
    frames = []
    for t in range(positions.shape[0]):
        if trunc:
            if np.any(np.abs(positions[t, :, :]) > 40):
                break
        else:
            frame = create_frame(t, n_particles, positions, velocities, sizes)
            frames.append(frame)

    gif.save(frames, output_path, duration=0.5, unit="s")


def load_simulations(path):
    if not path.endswith(".npy"):
        path = path + ".npy"
    full_arr = np.load(path)
    return full_arr
