import matplotlib.pyplot as plt
import numpy as np

def metres_per_second(u, v):
    initial_velocity = round(u * 0.44704, 3)
    final_velocity = round(v * 0.44704, 3)

    return print(f"{u} mph is {initial_velocity} metres per second\n"
                 f"{v} mph is {final_velocity} metres per second")


def calculate_distances(u: float, v: float, D: float,
                        n: int, thinking_time=0.0, use_thinking_distance=False) -> list:
    """
    This function calculates the position of the transverse bars
    Takes parameters;
    u = initial velocity in metres per second
    v = final velocity in metres per second
    D = total distance in metres
    n = number of bars
    thinking_time in seconds
    use_thinking_distance, set to false for no thinking time
    """
    distances = [0] * n

    x_0 = u * thinking_time
    a = (v**2 - u**2) / (2*D)
    t = (v - u) / a
    delta_t = t / (n-1)
    x_0_bars = thinking_time / delta_t

    # Calculate the distances for the first n_0 time intervals, where the speed is constant
    n_0 = int(round(x_0_bars)) # Number of time intervals during which the speed is constant

    for i in range(1, n):
        if use_thinking_distance:
            const_vel = [x_0 / n_0] * n_0
            for i in range(1, n_0):
                const_vel[i] += const_vel[i-1]
                distances[1:n_0+1] = const_vel
                # Calculate distances for remaining time intervals, where constant acceleration applies
            d_remaining = D - x_0
            a = (v**2 - u**2) / (2*d_remaining)
            t = (v - u) / a
            delta_t = t / (n-(n_0+1))

            for i in range(n_0, n):
                t_i = delta_t * (i - n_0) # Time since the end of constant velocity phase
                distances[i] = x_0 + (u * t_i) + (0.5 * a * (t_i ** 2))
                # Round distances to three decimal places
        else:
            distances = [a*(delta_t*k)**2/2 + u*(delta_t*k) for k in range(n)]

            #dk_values.insert(0, x_0) # insert x_0 at the beginning of the list
            #distances.append(D) # append d at the end of the list
    distances = [round(x, 3) for x in distances]
    return distances


def distance_dk(list: list) -> list:
    """
    Take a list of the bar markings in relation to x=0
    returns the distance between each bar
    :param list:
    :return: list
    """
    # Calculate the list of distances between consecutive markers
    distances = [list[0]] + [list[k] - list[k-1] for k in range(1, len(list))]
    distances = [round(x, 3) for x in distances]

    # We get rid of the zero element
    distances = distances[1:len(distances)]

    return distances

def plot_subgraphs_dk(no_thinking_time_list: list, thinking_time_list: list, real_world_list: list) -> plt:
    """
    Takes either;
        new_distances for n=90
        new_distances2 for n=45
    for real_world_list parameter
    :param no_thinking_time_list:
    :param thinking_time_list:
    :param real_world_list:
    :return: figure object containing two subplots
    """
        # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # No Thinking time
    n = len(no_thinking_time_list) + 1
    axs[0].plot(np.arange(1, n), no_thinking_time_list)
    axs[0].plot(np.arange(1, n), real_world_list)
    axs[0].set_title('No Thinking Time', fontsize='small')
    axs[0].set_xlabel('$d_k$')
    axs[0].set_ylabel('Distance between consecutive markers / m')
    x_ticks = np.arange(0, n, 10)
    axs[0].set_xticks(x_ticks)
    axs[0].legend(['Model', 'Real World'])

    # Thinking Time
    n = len(real_world_list) + 1
    axs[1].plot(np.arange(1, n), thinking_time_list)
    axs[1].plot(np.arange(1, n), real_world_list)
    axs[1].set_title('With Thinking Time', fontsize='small')
    axs[1].set_xlabel('$d_k$')
    axs[1].set_ylabel('Distance between consecutive markers / m')
    x_ticks = np.arange(0, n, 10)
    axs[1].set_xticks(x_ticks)
    axs[1].legend(['Model', 'Real World'])

    fig.subplots_adjust(wspace=0.3)
    fig.suptitle("Distance between each successive $d_k$", fontsize='large')

    filename = f"sidebyside_dk_n={n}.png"
    plt.savefig(filename)

    return plt.show()


def plot_graph_dk(model_list: list, real_world_list: list) -> plt:
    """
    Takes two lists for model and real-world distances and plots them on a single graph.
    :param model_list:
    :param real_world_list:
    :return: figure object containing a single plot
    """
    # Create figure with a single plot
    fig, ax = plt.subplots(figsize=(8, 6))

    n = len(model_list) + 1
    ax.plot(np.arange(1, n), model_list)
    ax.plot(np.arange(1, n), real_world_list)
    ax.set_title(f"Distance between each successive $d_k$\n$n={n}$", fontsize="large")
    ax.set_xlabel("$d_k$")
    ax.set_ylabel("Distance between consecutive markers / m")
    x_ticks = np.arange(0, n, 10)
    ax.set_xticks(x_ticks)
    ax.legend(["Model", "Real World"])

    filename = f"single_dk_n={n}.png"
    plt.savefig(filename)

    return plt.show()