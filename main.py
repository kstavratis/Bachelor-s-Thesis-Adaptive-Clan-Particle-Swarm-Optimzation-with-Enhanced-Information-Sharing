"""
File which is to be run from a command prompt.
The file can be run (from the root directory) likeso:
`python main.py -c -s 1234 configs/clan_base.json`


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
import argparse, json

from concurrent.futures.process import ProcessPoolExecutor

import numpy as np

from src.scripts.experiments.experiment import run

# Utility function for progress bar. https://github.com/tqdm/tqdm
# If the software is running on very low memory requirements, it may be removed as unnecessary.
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser(prog='Particle Swarm Optimization (PSO) experiments',
                                     description='Executes experiments of PSO as decreed by the input configuration file.')
    

    parser.add_argument('configuration_file_path', nargs='?', type=str, default='configs/classic_base.json')
    parser.add_argument('-c', '--concurrent', action='store_true',
                        help='Determine whether the experiments will be conducted in parallel (True)\
                        or in a single thread (False)\n\
                        Default value is `False`.'
                        )
    parser.add_argument('-s', '--seed', action='store', type=int, required=False,
                        help='A seed which determines the random initialization of any swarms created.\
                        The main purpose of this argument is to ensure a fair comparison (benchmarking)\
                        of the algorithms\
                        by having the particles of swarms start at identical initial positions.\
                        A secondary use is the reproducability of the results.\n\
                        Default value is `None`.') # Default value is `None`.
    args = parser.parse_args()
    filepath = args.configuration_file_path

    # Variable declarations
    data : dict = None

    with open(filepath) as config_file:
        data = json.load(config_file)

    nr_experiments : int = data['nr_experiments']
    # Seed generator for a single experiment
    rand_gen = np.random.default_rng(args.seed)
    infty = np.iinfo(np.int_).max 

    if args.concurrent:    
        executor = ProcessPoolExecutor()
        list(tqdm(
            executor.map(run,
                    [data] * nr_experiments,
                    rand_gen.integers(infty, size=nr_experiments)
                ),
            total=nr_experiments)
        )
    else:
        for _ in tqdm(range(nr_experiments)):
            run(data, rand_gen.integers(infty))


if __name__ == "__main__":
    main()
