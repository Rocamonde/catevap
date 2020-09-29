"""
This file is not meant to be a module, it's a dirty script that evolves the
traffic for a pre-set grid shape and evaluates the result of a custom loss
function.
"""

import numpy as np

import itertools
import pandas as pd
from grid_fixtures import grid3x3, grid6x6, grid12x6
from grid import Grid
from loss_function import evaluate_policy
from policies import Policy0, Policy1, Policy2
from vehicle import Vehicle
import datetime
import traceback

print('Welcome to the Analysis Script for the Traffic Model')
print('This script saves the results to a results.pkl file')
print(f'Began at {datetime.datetime.now()}')

# Independent Variables
policies = {
    'Policy0': Policy0,
    'Policy1': Policy1,
    'Policy2': Policy2
}
grid_fixtures = {
    'grid3x3': grid3x3,
    'grid6x6': grid6x6,
    # 'grid12x6': grid12x6
}
initial_speeds = range(1, Vehicle.max_speed+1, 4)
traffic_densities = np.arange(0.2, 1, 0.2)
epochs = np.linspace(10, 3000, 25).astype(np.uint64)

counter = 0
items = list(itertools.product(policies.items(), grid_fixtures.items(),
                               initial_speeds, traffic_densities, epochs))
total = len(items)

results = pd.DataFrame(columns=['policy', 'grid_fixture', 'initial_speed',
                                'traffic_density', 'epochs',
                                'A', 'B', 'C1', 'C2'],
                       index=np.arange(0, total))

for (pkey, policy), (gkey, grid_fixture), \
        initial_speed, traffic_density, epoch in items:
    counter += 1
    try:
        print(f"\rIteration {counter}. {int(counter/total*10000)/100}% completed", end="")
        eval_results = evaluate_policy(
            policy_cls=policy,
            lattice=Grid.new_from_arr(grid_fixture),
            initial_speed=initial_speed,
            traffic_density=traffic_density,
            epochs=epoch)
        A, B, C1, C2 = eval_results
        results.loc[counter-1] = [pkey, gkey, initial_speed, traffic_density,
                                  epoch, A, B, C1, C2]

        # We use checkpoints to partially save the progress
        if counter in list(np.linspace(1, total, 20).astype(int)):
            results.to_pickle('results.pkl')
            print("\nCheckpoint created.")

    except Exception as e:
        # If anything goes wrong for any reason, we print the error and move on
        # to the next case.
        print("An exception occurred")
        print(e)
        print(traceback.format_exc())

results.to_pickle('results.pkl')
print("\nCompleted!")
