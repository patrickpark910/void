"""
VOID COEFFICIENT (MCNP)

Written by Patrick Park (RO, Physics '22)
ppark@reed.edu

This proejct should be available at 
https://github.com/patrickpark910/void/

First written Feb. 15, 2021
Last updated Feb. 15, 2021

__________________
Default MCNP units

Length: cm
Mass: g
Energy & Temp.: MeV
Positive density (+x): atoms/barn-cm
Negative density (-x): g/cm3
Time: shakes
(1 barn = 10e-24 cm2, 1 sh = 10e-8 sec)

_______________
Technical Notes

"""

import os, sys, multiprocessing
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from mcnp_funcs import *

FILEPATH = os.path.dirname(os.path.abspath(__file__))
WATER_MAT_CARD = '102'
WATER_DENSITIES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # np.arange(start=0.1,stop=1.0,step=0.1)
# Prefer hardcoded lists rather than np.arange, which produces imprecise floating points, e.g., 0.7000000...003
INPUTS_FOLDER_NAME = 'inputs'

def main():
    initialize_rane()
    BASE_INPUT_NAME = 'void-a100-h100-r100.i' # find_base_file(FILEPATH)
    check_kcode(FILEPATH, BASE_INPUT_NAME)

    """
    print("The following lines will ask desired rod heights for this calculation.")
    
    rod_heights_dict = {}
    for rod in RODS:
        height = input(f"Input desired integer height for the {rod} rod: ")
        rod_heights_dict[rod] = height
    
    input_created = change_rod_height(FILEPATH, 'void', rod_heights_dict, BASE_INPUT_NAME, INPUTS_FOLDER_NAME)
    if input_created: print(f"Created {num_inputs_created} new input deck.")
    if not input_created: print(f"\n--Skipped {num_inputs_skipped} input deck because it already exists.")
    """
    num_inputs_created = 0
    num_inputs_skipped = 0
    for i in range(0, len(WATER_DENSITIES)):
        cell_densities_dict = {WATER_MAT_CARD: WATER_DENSITIES[i]}
        input_created = change_cell_densities(FILEPATH, 'void', cell_densities_dict, BASE_INPUT_NAME, INPUTS_FOLDER_NAME)
        if input_created: num_inputs_created += 1
        if not input_created: num_inputs_skipped += 1

    print(f"Created {num_inputs_created} new input decks.\n"
          f"--Skipped {num_inputs_skipped} input decks because they already exist.")

    if not check_run_mcnp(): sys.exit()

    # Run MCNP for all .i files in f".\{inputs_folder_name}".
    tasks = get_tasks()
    for file in os.listdir(f"{FILEPATH}/{INPUTS_FOLDER_NAME}"):
        run_mcnp(FILEPATH,f"{FILEPATH}/{INPUTS_FOLDER_NAME}/{file}",OUTPUTS_FOLDER_NAME,tasks)

    # Deletes MCNP runtape and source dist files.
    delete_files(f"{FILEPATH}/{OUTPUTS_FOLDER_NAME}",r=True,s=True)



if __name__ == '__main__':
    main()