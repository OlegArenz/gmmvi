import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

plt.ion()


def visualize_samples(samples, thinning=1, create_figure=True):
    if create_figure:
        plt.figure()
    [num_samples, num_dimensions] = samples.shape
    for i in range(0, num_samples, thinning):
        visualize_n_link(samples[i], num_dimensions, np.ones(num_dimensions))

def visualize_n_link(theta, num_dimensions, l, clear_fig=True):
    if clear_fig:
        plt.clf()
    plt.xlim([-0.2 * num_dimensions, num_dimensions])
    plt.ylim([-0.5 * num_dimensions, 0.5 * num_dimensions])

    x = [0]
    y = [0]
    for i in range(0, num_dimensions):
        y.append(y[-1] + l[i] * np.sin(np.sum(theta[:i + 1])))
        x.append(x[-1] + l[i] * np.cos(np.sum(theta[:i + 1])))
        plt.plot([x[-2], x[-1]], [y[-2], y[-1]], color='k', linestyle='-', linewidth=2)
    plt.plot(x[-1], y[-1], 'o')
    plt.plot(0.7 * num_dimensions, 0, 'rx')
    plt.pause(0.1)


def visualize_samples_multi(samples, num_goals, num_arms, num_links, thinning=1, create_figure=True, clear_fig=True):
    if create_figure:
        plt.figure()
    [num_samples, num_dimensions] = samples.shape
    for i in range(0, num_samples, thinning):
        visualize_n_link_multi(samples[i], np.ones(num_links), num_goals, num_arms, num_links, clear_fig=clear_fig)

def visualize_samples_multi_grid(samples, num_goals, num_arms, num_links, thinning=1, create_figure=True, clear_fig=True):

    [num_samples, num_dimensions] = samples.shape
    length = np.ones(num_links)
    ee_x_all = np.cos(2 * np.pi * np.arange(num_goals) / num_goals)
    ee_y_all = np.sin(2 * np.pi * np.arange(num_goals) / num_goals)
    if create_figure:
        fig, axs = plt.subplots(10, 10)
    for theta, ax in zip(samples, axs.flatten()):
        ax.set_xlim([-num_links, num_links])
        ax.set_ylim([-num_links, num_links])

        theta = np.reshape(theta, (num_arms, len(length)))

        x = [np.zeros(num_arms)]
        y = [np.zeros(num_arms)]
        for i in range(0, num_links):
            y.append(y[-1] + length[i] * np.sin(np.sum(theta[:, :i + 1], axis=1)))
            x.append(x[-1] + length[i] * np.cos(np.sum(theta[:, :i + 1], axis=1)))
            ax.plot([x[-2], x[-1]], [y[-2], y[-1]], color='k', linestyle='-', linewidth=2)
        ax.plot(x[-1], y[-1], 'o')

        for ee_x, ee_y in zip(ee_x_all, ee_y_all):
            ax.plot(0.7 * num_links * ee_x, 0.7 * num_links * ee_y, 'rx')
        # plt.plot(0.7 * num_dimensions, 0, 'rx')
        # plt.plot(- 0.7 * num_dimensions, 0, 'rx')
        # plt.plot(0, 0.7 * num_dimensions, 'rx')
        # plt.plot(0, - 0.7 * num_dimensions, 'rx')
        # if num_goals == 8:
        #     plt.plot(0.7 * num_dimensions / np.sqrt(2), 0.7 * num_dimensions / np.sqrt(2), 'rx')
        #     plt.plot(0.7 * num_dimensions / np.sqrt(2), - 0.7 * num_dimensions / np.sqrt(2), 'rx')
        #     plt.plot(- 0.7 * num_dimensions / np.sqrt(2), 0.7 * num_dimensions / np.sqrt(2), 'rx')
        #     plt.plot(- 0.7 * num_dimensions / np.sqrt(2), - 0.7 * num_dimensions / np.sqrt(2), 'rx')
        plt.pause(0.1)


def visualize_n_link_multi(theta, length, num_goals, num_arms, num_links, clear_fig=True):
    if clear_fig:
        plt.clf()
    plt.xlim([-num_links, num_links])
    plt.ylim([-num_links, num_links])

    theta = np.reshape(theta, (num_arms, len(length)))

    x = [np.zeros(num_arms)]
    y = [np.zeros(num_arms)]
    for i in range(0, num_links):
        y.append(y[-1] + length[i] * np.sin(np.sum(theta[:, :i + 1], axis=1)))
        x.append(x[-1] + length[i] * np.cos(np.sum(theta[:, :i + 1], axis=1)))
        plt.plot([x[-2], x[-1]], [y[-2], y[-1]], color='k', linestyle='-', linewidth=2)
    plt.plot(x[-1], y[-1], 'o')

    ee_x_all = np.cos(2 * np.pi * np.arange(num_goals) / num_goals)
    ee_y_all = np.sin(2 * np.pi * np.arange(num_goals) / num_goals)

    for ee_x, ee_y in zip(ee_x_all, ee_y_all):
        plt.plot(0.7 * num_links * ee_x, 0.7 * num_links * ee_y, 'rx')
    # plt.plot(0.7 * num_dimensions, 0, 'rx')
    # plt.plot(- 0.7 * num_dimensions, 0, 'rx')
    # plt.plot(0, 0.7 * num_dimensions, 'rx')
    # plt.plot(0, - 0.7 * num_dimensions, 'rx')
    # if num_goals == 8:
    #     plt.plot(0.7 * num_dimensions / np.sqrt(2), 0.7 * num_dimensions / np.sqrt(2), 'rx')
    #     plt.plot(0.7 * num_dimensions / np.sqrt(2), - 0.7 * num_dimensions / np.sqrt(2), 'rx')
    #     plt.plot(- 0.7 * num_dimensions / np.sqrt(2), 0.7 * num_dimensions / np.sqrt(2), 'rx')
    #     plt.plot(- 0.7 * num_dimensions / np.sqrt(2), - 0.7 * num_dimensions / np.sqrt(2), 'rx')
    plt.pause(0.1)


def visualize_mixture(mixture_weights, mixture_means, l=None, clear_fig=True, markerPoses=[], create_figure=True,
                      comp_colors=None):
    num_dimensions = len(mixture_means[0])
    if l is None:
        l = np.ones(num_dimensions)
    if clear_fig:
        plt.clf()

    plt.xlim([-0.2 * num_dimensions, num_dimensions])
    plt.ylim([-0.5 * num_dimensions, 0.5 * num_dimensions])
    plt.xlim([-num_dimensions, num_dimensions])
    plt.ylim([-num_dimensions, num_dimensions])
    if np.max(mixture_weights) - np.min(mixture_weights) != 0:
        weights = mixture_weights - np.min(mixture_weights)
        weights = 0.1 + 0.9 * weights / (np.max(weights) - np.min(weights))
    else:
        weights = np.ones((len(mixture_weights)))

    if comp_colors is None:
        comp_colors = np.repeat("k", [len(weights)])
    for i in range(len(weights)):
        x = [0]
        y = [0]
        for j in range(0, num_dimensions):
            y.append(y[-1] + l[j] * np.sin(np.sum(mixture_means[i][:j + 1])))
            x.append(x[-1] + l[j] * np.cos(np.sum(mixture_means[i][:j + 1])))
            plt.plot([x[-2], x[-1]], [y[-2], y[-1]], color=comp_colors[i], linestyle='-', linewidth=2, alpha=weights[i], markersize=3)
        plt.plot(x[-1], y[-1], 'o', color="k", alpha=weights[i], markersize=6.1)
        plt.plot(x[-1], y[-1], 'o', color="red", alpha=weights[i], markersize=6)
    rect = patches.Rectangle((-0.25, -0.25), 0.5, 0.5, linewidth=1, edgecolor='k', fill=True, facecolor='dimgrey',
                             zorder=1000)
    ax = plt.gca()
    ax.add_patch(rect)
    [plt.plot(pose[0], pose[1], 'rx', markersize=10, mew=2) for pose in markerPoses]
   # plt.pause(0.5)
