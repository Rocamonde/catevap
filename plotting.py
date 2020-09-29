import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import itertools
from scipy.optimize import curve_fit

# plt.rcParams["font.size"] = 13
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

from vehicle import Vehicle

def fitline(x, m, n):
    return m*x+n

initial_speeds = range(1, Vehicle.max_speed+1, 4)
traffic_densities = np.arange(0.2, 1, 0.2)
grid_labels = {'grid3x3': '3x3', 'grid6x6': '6x6'}
df = pd.read_pickle('results.pkl').dropna()

policies = {'Policy0': r'\Pi_0', 'Policy1': r'\Pi_1', 'Policy2': r'\Pi_2'}
vars = {'A': 'A', 'B': 'B', 'C1': 'C_1', 'C2': 'C_2', 'U': 'U'}
velocities = [1, 5, 9, 13]

def plot_per_policy():
    # Per-policy plotting
    for policy, var in itertools.product(policies.keys(), vars.keys()):
        fig = plt.figure(figsize=(5, 3))
        for i, grid in enumerate(grid_labels.keys()):
            dfs = df[(df['policy'] == policy) & (df['grid_fixture'] == grid)]
            ax: Axes = plt.subplot(1, 2, i+1)

            pos1 = ax.get_position()  # get the original position
            pos2 = [pos1.x0, pos1.y0 + pos1.height*0.15, pos1.width, pos1.height *0.85]
            ax.set_position(pos2)  # set a new position
            ax.set_xlabel('Time ($t$ iterations)')

            if var == 'U':
                dfs['U'] = dfs['A']**2 + (dfs['B']/dfs['epochs']**2)**2 + (1-dfs['C1'])**2 + dfs['C2']**2
            elif var=='B':
                dfs['B'] = dfs['B'] / dfs['epochs'] ** 2
            for speed in initial_speeds:
                slice = dfs[(dfs['initial_speed'] == speed)]
                d = np.array(slice[slice['epochs'] > 75][['epochs', var]]).transpose()
                ax.scatter(d[0], d[1], label=r'$v_{\mathrm{max}}='+str(speed)+r'$',marker='x', s=20, linewidth=0.75)
                ax.set_title(f'{grid_labels[grid]} lattice')

        handles, labels = ax.get_legend_handles_labels()
        lgd = fig.legend(handles, labels, frameon=False, loc='lower center', ncol=4, handletextpad=0.1)
        fig.savefig(f'figures/{policy}-{var}.pdf', dpi=300, bbox_extra_artists=(lgd,))


def plot_per_var():
    # Comparing policies for different variables
    for var in vars.keys():
        fig = plt.figure(figsize=(5, 4))
        ax: Axes = plt.subplot(111)
        pos1 = ax.get_position()  # get the original position
        pos2 = [pos1.x0, pos1.y0 + pos1.height * 0.3, pos1.width, pos1.height * 0.7]
        ax.set_position(pos2)  # set a new position
        ax.set_ylabel(f'${vars[var]}$')
        ax.set_xlabel('Time ($t$) iterations')
        # ax.set_yscale('log')
        # ax.set_xscale('log')

        for policy, policy_latex in policies.items():
            dfs = df[(df['policy'] == policy) & (df['grid_fixture'] == 'grid6x6') & (df['initial_speed']  == 1)]
            if var == 'U':
                dfs['U'] = dfs['A']**2 + (dfs['B']/dfs['epochs']**2)**2 + (1-dfs['C1'])**2 + dfs['C2']**2
            dfs = dfs[dfs[var] < dfs[var].quantile(.95)]
            d = np.array(dfs[dfs['epochs'] > 75][['epochs', var]]).transpose()
            # Dirty fix due to the bug in loss function
            if var == 'B':
                d[1] = d[1] / d[0]**2
            ax.scatter(d[0], d[1], label=f'${policy_latex}$', marker='x', s=20, linewidth=0.75)


            try:
                popt, pcov = curve_fit(fitline, d[0], d[1])
                xvals = np.linspace(76, 3000, 500)
                ax.plot(xvals, fitline(xvals, *popt), label=f'Fit of ${policy_latex}$')
            except:
                pass

        ax.set_title(f'${vars[var]}$')
        handles, labels = ax.get_legend_handles_labels()
        lgd = fig.legend(handles, labels, frameon=False, loc='lower center', ncol=3, handletextpad=0.1)
        fig.savefig(f'figures/var-{var}.pdf', dpi=300, bbox_extra_artists=(lgd,))


def plot_var_per_velocity(var):
    for v in velocities:
        fig = plt.figure(figsize=(5, 3))
        ax: Axes = plt.subplot(111)
        pos1 = ax.get_position()  # get the original position
        pos2 = [pos1.x0, pos1.y0 + pos1.height * 0.15, pos1.width, pos1.height * 0.85]
        ax.set_position(pos2)  # set a new position

        for policy, policy_latex in policies.items():
            dfs = df[(df['policy'] == policy) & (df['grid_fixture'] == 'grid6x6') & (df['initial_speed'] == v)].copy()
            dfs['U'] = dfs['A'] ** 2 + (dfs['B'] / 10 ** 11) ** 2 + (1 - dfs['C1']) ** 2 + dfs['C2'] ** 2
            dfs = dfs[dfs[var] < dfs[var].quantile(.95)]
            d = np.array(dfs[dfs['epochs'] > 75][['epochs', 'U']]).transpose()
            ax.scatter(d[0], d[1], label=f'${policy_latex}$', marker='x', s=20, linewidth=0.75)

            try:
                popt, pcov = curve_fit(fitline, d[0], d[1])
                xvals = np.linspace(76, 3000, 500)
                ax.plot(xvals, fitline(xvals, *popt), label=f'Fit of ${policy_latex}$')
            except:
                pass

        # ax.set_title(f'{var} variable')
        handles, labels = ax.get_legend_handles_labels()
        lgd = fig.legend(handles, labels, frameon=False, loc='lower center', ncol=3, handletextpad=0.1)
        fig.savefig(f'figures/var-{var}-velocity-{v}.pdf', dpi=300, bbox_extra_artists=(lgd,))


def plot_var_per_density(var):
    for policy in policies.keys():
        fig = plt.figure(figsize=(5, 3))
        dfs = df[(df['policy'] == policy) & (df['grid_fixture'] == 'grid6x6')  & (df['initial_speed'] == 1)]
        ax: Axes = plt.subplot(111)

        pos1 = ax.get_position()  # get the original position
        pos2 = [pos1.x0, pos1.y0 + pos1.height*0.15, pos1.width, pos1.height *0.85]
        ax.set_position(pos2)  # set a new position
        dfs['B'] = dfs['B'] / dfs['epochs'] ** 2
        if var == 'U':
            dfs['U'] = dfs['A']**2 + (dfs['B'])**2 + (1-dfs['C1'])**2 + dfs['C2']**2

        for density in traffic_densities:
            slice = dfs[(dfs['traffic_density'] == density)]
            d = np.array(slice[slice['epochs'] > 50][['epochs', var]]).transpose()
            ax.scatter(d[0], d[1], label=r'$\rho='+str(int(density*10)/10)+r'$',marker='x', s=20, linewidth=0.75)

        handles, labels = ax.get_legend_handles_labels()
        lgd = fig.legend(handles, labels, frameon=False, loc='lower center', ncol=4, handletextpad=0.1)
        fig.savefig(f'figures/density-{policy}-{var}.pdf', dpi=300, bbox_extra_artists=(lgd,))


if __name__ == '__main__':
    # plot_per_policy()
    # plot_per_var()
    # plot_u_per_velocity()
    plot_var_per_density('B')