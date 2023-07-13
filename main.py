"""
Code to conduct experiments with the Particle Swarm Optimization (PSO) and some of its variations (including ClanPSO, Adaptive PSO, EIS PSO)
Code for my Bachelor's Thesis in the Informatics department of the Aristotle University of Thessaloniki (https://www.csd.auth.gr/en/)
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

from src.scripts.experiments.experiment import run

# Utility function for progress bar in the case of sequential execution. https://github.com/tqdm/tqdm
# If the software is running on very low memory requirements, it may be removed as unnecessary.
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser(prog='Particle Swarm Optimization (PSO) experiments',
                                     description='Executes experiments of PSO as decreed by the input configuration file.')
    

    parser.add_argument('configuration_file_path', nargs='?', type=str, default='configs/clan_base.json')
    parser.add_argument('-c', '--concurrent', action='store_true',
                        help='Determine whether the experiments will be conducted in parallel (True) or in a single thread (False) (default : False)'
                        )
    args = parser.parse_args()
    filepath = args.configuration_file_path

    # Variable declarations
    data : dict = None

    with open(filepath) as config_file:
        data = json.load(config_file)

    nr_experiments : int = data['nr_experiments']

    if args.concurrent:    
        executor = ProcessPoolExecutor()
        list(tqdm(executor.map(run, [data] * nr_experiments), total=nr_experiments))
    else:
        for _ in tqdm(range(nr_experiments)):
            run(data)


if __name__ == "__main__":
    main()
