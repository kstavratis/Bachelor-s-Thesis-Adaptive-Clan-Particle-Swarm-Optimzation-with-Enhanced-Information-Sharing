"""
Copyright (C) 2023  Konstantinos Stavratis
e-mail: kostauratis@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os, glob, json
from functools import reduce
from multiprocessing import BoundedSemaphore

import pandas as pd

# The log_semaphore is used so as to disallow two directories being named with the same (number) ID,
# resulting in one write overwriting the previous experiment.
# NOTE: The existence of the semaphore is of significance only in the case of running experiments concurrently.
log_semaphore = BoundedSemaphore(1)

def log_pso(config_data : dict, log_names : list[str], log_lists : list[list[pd.DataFrame]]):
    """
    Stores pso experimental results provided in the `/experiments` directory (which it creates if it doesn't already exist) in .csv file format.
    In general, the full path created for the provided data is of the form:
    `/experiments/topology/objective_function/particles/MAX+1/`
    where `MAX` is the folder whose name is the maximum integer before the call of this function.

    This function supports concurrent writing.

    Parameters
    ----------
    config_data : dict
        A dictionary containing keywords required to create a path.
        With the current implementation, the fields required from this dictionary are: "classes", "topology", "objective_function" and "nr_particles".

    log_names : Iterable[str] (tuple or list)
        An (ordered) set of strings, each element of which will be the names of the .csv log files that will be created.
    
    log_lists : Iterable[Iterable[pd.DataFrame]]
        An (ordered) set of data which will be logged in a .csv file.
    """

    # Dynamic sanity check
    assert len(log_names) == len(log_lists), f'{len(log_names)} != {len(log_lists)}. However, the `log_names` and `log_lists` arguments must have the same length.'

    store_path = '' # Variable declaration

    if 'clans' in config_data:
        store_path = __get_clan_pso_filepath(config_data)
    else:
        store_path = __get_backbone_pso_filepath(config_data)

    os.makedirs(store_path, exist_ok=True)

    log_semaphore.acquire()

    current_numeral_folders = [os.path.basename(filename) for filename in glob.glob(f'{store_path}/[0-9]*')]
    experiment_id = '' # Variable declaration

    if not current_numeral_folders: # If no experiment has been conducted...
        experiment_id = '1'
    else:
        # Assign as value the next integer to the current maximum value (i.e. maximum of folder_id + 1)
        experiment_id = str(reduce(max, [int(folder_id) for folder_id in current_numeral_folders]) + 1)

    store_path = os.sep.join((store_path, experiment_id))
    os.makedirs(store_path)

    log_semaphore.release()


    
    # Make a copy of the configuration JSON file, so that the configuration of the experiment is known
    with open(os.path.join(store_path, 'config.json'), 'w') as config_copy:
        config_object = json.dumps(config_data, indent=4)
        config_copy.write(config_object) 

    
    for log_name, log_list in zip(log_names, log_lists):
        # Store the positions of the particles for the whole algorithm.
        pd.concat(log_list).to_csv(os.path.join(store_path, f'{log_name}.csv'))

    print(f'Logged files for experiment #{experiment_id}')


def __get_backbone_pso_filepath(config_data : dict) -> str:
    pso_type_subpath = ''
    for c in config_data['classes']:
        pso_type_subpath = '_'.join((pso_type_subpath, c))

    store_path = f'experiments/{config_data["topology"]}/{config_data["objective_function"]}/{pso_type_subpath}/particles{config_data["nr_particles"]}'

    return store_path

def __get_clan_pso_filepath(config_data : dict) -> str:
    
    nr_clans = len(config_data['clans'])

    #! WARNING: The current implementation assumes that all clans have the same number of particles and have identical behaviour.
    #! The conference of leaders is also assumed to behave identically to the clans.
    #! This is an ad-hoc solution for logging. However, the codebase itself supports different behaviour of each component (i.e. individual clans and conference).
    #! NOTE: This is to make for much easier and shorter folder names.
    #! This can be easily modified to capture the full information
    first_clan = config_data['clans'][0] # The first clan will act as a behaviour representative of the whole configuration file. See above WARNING message.

    nr_particles_per_clan = first_clan['nr_particles']

    pso_type_subpath = '' # Variable declaration
    for c in first_clan['classes']:
        pso_type_subpath = '_'.join((pso_type_subpath, c))
    
    store_path = f'experiments/{config_data["topology"]}/{config_data["objective_function"]}/{pso_type_subpath}/{nr_clans}x{nr_particles_per_clan}'

    return store_path

