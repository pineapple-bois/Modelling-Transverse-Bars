import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    fig.suptitle("Distance $d_k$ between each successive $D_n$", fontsize='large')

    #filename = f"sidebyside_dk_n={n}.png"
    #plt.savefig(filename)

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
    ax.set_title(f"Distance $d_k$ between each successive $D_n$\n$k={n}$", fontsize="large")
    ax.set_xlabel("$d_k$")
    ax.set_ylabel("Distance between consecutive markers / m")
    x_ticks = np.arange(0, n, 10)
    ax.set_xticks(x_ticks)
    ax.legend(["Model", "Real World"])

    #filename = f"single_dk_n={n}.png"
    #plt.savefig(filename)

    return plt.show()

def plot_single_dk(model_list: list) -> plt:
    """
    :param model_list:
    :return: figure object containing a single plot
    """
    # Create figure with a single plot
    n = len(model_list)
    plt.plot(np.arange(1, n+1), model_list)
    plt.title(f"Distance $d_k$ between each successive $D_n$\n$k={n}$", fontsize="large")
    plt.xlabel("$d_k$")
    plt.ylabel("Distance between consecutive markers / m")
    x_ticks = np.arange(0, n+1, 5)
    plt.xticks(x_ticks)
    #filename = f"single_dk_n={n}.png"
    #plt.savefig(filename)

    return plt.show()


def get_delta_t(u: float, v: float, D: float,
                n: int, thinking_time=0.0, use_thinking_distance=False):
    """
    Takes similar parameters to calculate_distances function
    returns delta_t's
    delta_t2 = 0.0 if use_thinking_distance = default = False
    """
    x_0 = u * thinking_time
    a = (v**2 - u**2) / (2*D)
    t = (v - u) / a
    delta_t1 = t / (n-1)
    x_0_bars = thinking_time / delta_t1
    n_0 = int(round(x_0_bars))

    if use_thinking_distance:
        d_remaining = D - x_0
        a2 = (v**2 - u**2) / (2*d_remaining)
        t2 = (v - u) / a2
        delta_t2 = t2 / (n-(n_0+1))
    else:
        delta_t2 = 0.0

    delta_t1 = round(delta_t1, 6)
    delta_t2 = round(delta_t2, 6)

    print(f"Delta_t1: {delta_t1}\nDelta_t2: {delta_t2}")
    return delta_t1, delta_t2


def distance_time_graph(u: float, v: float, D: float, n: int,
                 thinking_time=0.0, use_thinking_distance=False):
    """
        Takes parameters;
    u = initial velocity in metres per second
    v = final velocity in metres per second
    D = total distance in metres
    n = number of bars
    thinking_time in seconds
    use_thinking_distance, set to false for no thinking time
    delta_t2 = 0.0 if use_thinking_distance = default = False

    returns list of 2D vectors
    """
    distances_list = calculate_distances(u, v, D, n, thinking_time, use_thinking_distance)
    graph_points = []

    n = len(distances_list)
    x_0 = u * thinking_time
    a = (v**2 - u**2) / (2*D)
    t = (v - u) / a
    delta_t1 = t / (n-1)
    x_0_bars = thinking_time / delta_t1
    n_0 = int(round(x_0_bars))

    if use_thinking_distance:
        d_remaining = D - x_0
        a2 = (v**2 - u**2) / (2*d_remaining)
        t2 = (v - u) / a2
        delta_t2 = t2 / (n-(n_0+1))
    else:
        delta_t2 = 0.0

    for i in range(n):
        if i <= n_0:
            t_i = round(i * delta_t1, 3)
        else:
            if use_thinking_distance:
                t_i = round((n_0 * delta_t1) + ((i - n_0) * delta_t2), 3)
            else:
                t_i = round(i * delta_t1, 3)
        graph_points.append([t_i, distances_list[i]])
        if i == n_0 and use_thinking_distance:
            delta_t1 = delta_t2

    # Extract the time and distance values from the graph points
    times = [point[0] for point in graph_points]
    distances = [point[1] for point in graph_points]

    if use_thinking_distance:
        string = "With Thinking Time"
    else:
        string = "No Thinking Time"
    # Plot the graph using the time and distance values
    plt.plot(times, distances)
    plt.title(f"Distance/Time Graph\n{string}, $n={n}$")
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')

    return plt.show(), print(len(graph_points), graph_points)


def velocity_time_graph(u: float, v: float, D: float, n: int, thinking_time=0.0, use_thinking_distance=False):
    """
    This function creates a velocity time graph
    Takes parameters;
    u = initial velocity in metres per second
    v = final velocity in metres per second
    D = total distance in metres
    n = number of bars
    thinking_time in seconds
    use_thinking_distance, set to false for no thinking time

    Returns velocity/time graph
    """

    # Equations of motion
    a = (v**2 - u**2) / (2*D)
    t = (v - u) / a
    delta_t1 = t / (n-1)
    x_0 = u * thinking_time
    x_0_bars = thinking_time / delta_t1
    n_0 = int(round(x_0_bars))

    for i in range(1,n):
        if use_thinking_distance:
            d_remaining = D - x_0
            a2 = (v ** 2 - u ** 2) / (2 * d_remaining)
            t2 = (v - u) / a2
            delta_t2 = t2 / (n - (n_0 + 1))
            v_list_const = [u for n in range(0,n_0)]
            v_list_acc = [u + a * delta_t2 * n for n in range(n_0, n)]
            vn_values = v_list_const + v_list_acc
        else:
            vn_values = [u + a * delta_t1 * n for n in range(0, n)]

    vn_values = [round(x, 3) for x in vn_values]

    num_points = len(vn_values)
    if use_thinking_distance:
        time_list = [round(delta_t1 * i, 3) if i <= n_0 else round(delta_t2 * i, 3)for i in range(n)]
    else:
        time_list = [delta_t1 * n for n in range(num_points)]


    # plot d over time
    if use_thinking_distance:
        string = f"With Thinking Time, $n={n}$"
    else:
        string = f"No Thinking Time, $n={n}$"
    plt.plot(time_list, vn_values)
    plt.xlabel('Time / s')
    plt.ylabel('Velocity / $\mathrm{m\ s{^{-1}}}$')
    plt.title(f'Velocity vs. Time\n{string}')
    if use_thinking_distance:
        title = f"vel_time_thinking_n={n}"
    else:
        title = f"vel_time_no_thinking_n={n}"

    #plt.savefig(f'{title}.png')

    return plt.show()


def dataframe(real_world_list: list, thinking_list: list, no_thinking_list: list):
    """

    """

    d_k = [f"d{i}" for i in range(1, len(real_world_list)+1)]
    dk_df = pd.DataFrame({'d_k': d_k, 'real_world': real_world_list,
                          'thinking': thinking_list, 'no_thinking': no_thinking_list})

    return dk_df


def stats(Dataframe):
    """
    Take a pandas DataFrame as input

    returns: summary statistics and boxplot
    """
    stats = Dataframe.describe()

    Dataframe.boxplot(column=['real_world', 'thinking', 'no_thinking'])
    plt.title("Box plot of real world vs. model data")
    plt.ylabel("Distance / metres")

    if len(Dataframe['real_world'].tolist()) > 45:
        string = "90 Transverse bars"
    else:
        string = "45 Transverse bars"

    return print(string), plt.show(), print(f"\nSummary Statistics:\n\n{stats}")