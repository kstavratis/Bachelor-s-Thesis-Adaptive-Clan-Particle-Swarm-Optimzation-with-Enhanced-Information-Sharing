"""
This file is responsible for the main loop of Particle Swarm Optimization (PSO).

Copyright (C) 2024  Konstantinos Stavratis
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


import importlib, functools

from src.scripts.benchmark_functions import benchmark_functions as bf
from src.classes.PSOs.pso_backbone import PSOBackbone
from src.classes.PSOs.clan_pso import ClanPSO


def handle_config_file_data(data):

    topology = data['topology']
    if topology == 'classic':
        return __handle_pso_backbone_config_data(data)
    elif topology == 'clan':
        return __handle_clan_config_file(data)
    else:
        raise NotImplementedError(f'Topology {topology} has not been implemented. Currently implemented topologies are ["classic", "clan"].')


def __handle_pso_backbone_config_data(data : dict):
    """
    Parameters
    ----------
    data : dict
        A dictionary containing all key-value pairs which will be used for the constructors of the PSO variation.

    Returns
    -------
    : PSOcombination
        A novel Particle Swarm Optimization (PSO) variation whose behaviour
        is the combination of the variations provided in the configuration file.\n
        WARNING: The order in which each variation is inserted in the configuration file *matters*
        for the behaviour!
    """

    nr_particles, nr_dimensions = int(data['nr_particles']), int(data['nr_dimensions'])
    # Get dictionary of the particular objection function defined inside `benchmark_functions` module.
    objective_function = getattr(bf, f'{data["objective_function"]}')

    # Variable declarations
    classes_pointers = []
    kwargs = {}

    for key in data['classes']:
        pso_dict = data['classes'][key]

        # module_path, class_name
        module_path, class_name = pso_dict['class_path'].rsplit('.', 1)
        pso_variation_module = importlib.import_module(module_path)
        classes_pointers.append(getattr(pso_variation_module, class_name))

        kwargs.update(pso_dict['kwargs'])


    # Adding `PSOBackbone` properties
    classes_pointers.append(PSOBackbone)
    kwargs.update(
        {
            'nr_particles' : nr_particles, 'nr_dimensions' : nr_dimensions,
            'objective_function' : objective_function['formula'],
            'domain_boundaries' : objective_function['search_domain']
        }
    )

    # DYNAMICALLY construct the PSO variation based on the configuration file.
    # This is the "revolutionary" part of this framework!
    class PSOcombination(*classes_pointers):
        def __init__(self, kwargs):
            super().__init__(**kwargs)

    return PSOcombination(kwargs)


def __handle_clan_config_file(data : dict):

    clans = []

    for c in data['clans']:
        clans.append(__handle_pso_backbone_config_data({**c, **{'objective_function': data['objective_function'], 'nr_dimensions' : data['nr_dimensions'] }}))
        # The "appended" dictionary consists of the values which are shared among all "classic" PSOs.

    clan_pso_instance = ClanPSO(clans)

    # Variable declarations
    conference_behaviour_classes_pointers = []
    conference_behaviour_kwargs = {}

    for key in data['conference_behaviour']['classes']:
        pso_dict = data['conference_behaviour']['classes'][key]
        pso_variation_module = importlib.import_module(pso_dict['module_path'])
        conference_behaviour_classes_pointers.append(getattr(pso_variation_module, pso_dict['class_name']))

        conference_behaviour_kwargs.update(pso_dict['kwargs'])


    # Adding `PSOBackbone` properties
    objective_function = getattr(bf, f'{data["objective_function"]}')
    
    conference_behaviour_classes_pointers.append(PSOBackbone)
    conference_behaviour_kwargs.update(
        {
            'nr_particles' : 1, 'nr_dimensions' : 1, # These will be ignored, as clans only take into consideration the already formed clans.
            'objective_function' : objective_function['formula'],
            'domain_boundaries' : objective_function['search_domain']
        }
    )

    # Fixing the stepping behaviour of the clan swarm in all iterations.
    #! WARNING: This may go contrary to the initial vision
    #! of allowing the user to determine the behaviour of the conference of the leaders at each individual step.
    #! This part needs to be removed and its dependencies altered to achieve the aforementioned vision.
    clan_pso_instance.step = functools.partial(clan_pso_instance.step, conference_behaviour_classes_pointers, conference_behaviour_kwargs)

    return clan_pso_instance # , conference_behaviour_classes_pointers, conference_behaviour_kwargs