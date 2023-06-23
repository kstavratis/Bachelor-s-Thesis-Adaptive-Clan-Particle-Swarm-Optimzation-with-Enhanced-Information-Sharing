import importlib, functools

import src.scripts.benchmark_functions as bf
from src.classes.PSOs.pso_backbone import PSOBackbone
from src.classes.PSOs.clan_pso import ClanPSO


def handle_config_file_data(data):

    topology = data['topology']
    if topology == 'classic':
        return __handle_pso_backbone_config_data(data)
    elif topology == 'clan':
        return __handle_clan_config_file(data)
    else:
        return NotImplementedError(f'Topology {topology} has not been implemented. Currently implemented topologies are ["classic", "clan"].')


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
    
    : class
        A pointer to the generated class representing the novel PSO variation built by the configuration file.
    """

    nr_particles, nr_dimensions = int(data['nr_particles']), int(data['nr_dimensions'])
    # Get dictionary of the particular objection function defined inside `benchmark_functions` module.
    objective_function = getattr(bf, f'{data["objective_function"]}' + '_function')

    # Variable declarations
    classes_pointers = []
    kwargs = {}

    for key in data['classes']:
        pso_dict = data['classes'][key]
        pso_variation_module = importlib.import_module(pso_dict['module_path'])
        classes_pointers.append(getattr(pso_variation_module, pso_dict['class_name']))

        kwargs |= pso_dict['kwargs']


    # Adding `PSOBackbone` properties
    classes_pointers.append(PSOBackbone)
    kwargs |= {
        'nr_particles' : nr_particles, 'nr_dimensions' : nr_dimensions,
        'objective_function' : objective_function['formula'],
        'domain_boundaries' : objective_function['search_domain']
    }

    # DYNAMICALLY construct the PSO variation based on the configuration file.
    class PSOcombination(*classes_pointers):
        def __init__(self, kwargs):
            super().__init__(**kwargs)

    return PSOcombination(kwargs) # , PSOcombination, kwargs


def __handle_clan_config_file(data : dict):

    clans = []

    for c in data['clans']:
        clans.append(__handle_pso_backbone_config_data(c | {'objective_function': data['objective_function'] }))
        # Permits to write the `objective_value` field only once in the JSON configuration file.

    clan_pso_instance = ClanPSO(clans)

    # Variable declarations
    conference_behaviour_classes_pointers = []
    conference_behaviour_kwargs = {}

    for key in data['conference_behaviour']['classes']:
        pso_dict = data['conference_behaviour']['classes'][key]
        pso_variation_module = importlib.import_module(pso_dict['module_path'])
        conference_behaviour_classes_pointers.append(getattr(pso_variation_module, pso_dict['class_name']))

        conference_behaviour_kwargs |= pso_dict['kwargs']


    # Adding `PSOBackbone` properties
    objective_function = getattr(bf, f'{data["objective_function"]}' + '_function')
    conference_behaviour_classes_pointers.append(PSOBackbone)
    conference_behaviour_kwargs |= {
        'nr_particles' : 1, 'nr_dimensions' : 1, # These will be ignored, as clans only take into consideration the already formed clans.
        'objective_function' : objective_function['formula'],
        'domain_boundaries' : objective_function['search_domain']
    }

    # Fixing the stepping behaviour of the clan swarm in all iterations.
    #! WARNING:
    #! This may go contrary to the initial vision of allowing the user to determine the behaviour of the conference of the leaders
    #! at each individual step.
    #! This part needs to be removed and its dependencies altered to achieve the aforementioned vision.
    clan_pso_instance.step = functools.partial(clan_pso_instance.step, conference_behaviour_classes_pointers, conference_behaviour_kwargs)

    return clan_pso_instance # , conference_behaviour_classes_pointers, conference_behaviour_kwargs