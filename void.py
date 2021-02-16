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
OUTPUTS_FOLDER_NAME = 'outputs'
MODULE_NAME = 'void'
KEFF_CSV_NAME = f'{MODULE_NAME}_keff.csv'
RHO_CSV_NAME = f'{MODULE_NAME}_rho.csv'
PARAMS_CSV_NAME = f'{MODULE_NAME}_parameters.csv'
FIGURE_NAME = f'{MODULE_NAME}_results.png'


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
    
    input_created = change_rod_height(FILEPATH, MODULE_NAME, rod_heights_dict, BASE_INPUT_NAME, INPUTS_FOLDER_NAME)
    if input_created: print(f"Created {num_inputs_created} new input deck.")
    if not input_created: print(f"\n--Skipped {num_inputs_skipped} input deck because it already exists.")
    """

    num_inputs_created = 0
    num_inputs_skipped = 0
    for i in range(0, len(WATER_DENSITIES)):
        cell_densities_dict = {WATER_MAT_CARD: WATER_DENSITIES[i]}
        input_created = change_cell_densities(FILEPATH, MODULE_NAME, cell_densities_dict, BASE_INPUT_NAME, INPUTS_FOLDER_NAME)
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

    # Setup a dataframe to collect keff values
    keff_df = pd.DataFrame(columns=["density", "keff", "keff unc"]) # use lower cases to match 'rods' def above
    keff_df["density"] = WATER_DENSITIES
    keff_df.set_index("density",inplace=True)

    for water_density in WATER_DENSITIES:
        keff, keff_unc = extract_keff(f"{FILEPATH}/{OUTPUTS_FOLDER_NAME}/o_{MODULE_NAME}-m{WATER_MAT_CARD}-{''.join(c for c in str(water_density) if c not in '.')}.o")
        keff_df.loc[water_density, 'keff'] = keff
        keff_df.loc[water_density, 'keff unc'] = keff_unc

    print(f"\nDataframe of keff values and their uncertainties:\n{keff_df}\n")
    keff_df.to_csv(KEFF_CSV_NAME)

    convert_keff_to_rho_void(KEFF_CSV_NAME, RHO_CSV_NAME)
    # calc_params_void(RHO_CSV_NAME, PARAMS_CSV_NAME)
    # plot_data_void(KEFF_CSV_NAME, RHO_CSV_NAME, FIGURE_NAME)

    print(f"\n************************ PROGRAM COMPLETE ************************\n")


'''
Converts a CSV of keff and uncertainty values to a CSV of rho and uncertainty values.

keff_csv_name: str, name of CSV of keff values, including extension, "keff.csv"
rho_csv_name: str, desired name of CSV of rho values, including extension, "rho.csv"

Does not return anything. Only makes the actual file changes.
'''
def convert_keff_to_rho_void(keff_csv_name,rho_csv_name):
    # Assumes the keff.csv has columns labeled "rod" and "rod unc" for keff and keff uncertainty values for a given rod
    keff_df = pd.read_csv(keff_csv_name,index_col=0)
    rods = [c for c in keff_df.columns.values.tolist() if "unc" not in c]
    water_densities = keff_df.index.values.tolist()

    # Setup a dataframe to collect rho values
    rho_df = pd.DataFrame(columns=keff_df.columns.values.tolist()) # use lower cases to match 'rods' def above
    rho_df["density"] = water_densities
    rho_df.set_index("density",inplace=True)

    '''
    ERROR PROPAGATION FORMULAE
    % Delta rho = 100* frac{k2-k1}{k2*k1}
    numerator = k2-k1
    delta num = sqrt{(delta k2)^2 + (delta k1)^2}
    denominator = k2*k1
    delta denom = k2*k1*sqrt{(frac{delta k2}{k2})^2 + (frac{delta k1}{k1})^2}
    delta % Delta rho = 100*sqrt{(frac{delta num}{num})^2 + (frac{delta denom}{denom})^2}
    '''

    for water_density in water_densities:
        k1 = keff_df.loc[water_density,'keff']
        k2 = keff_df.loc[water_density[-1],'keff']
        dk1 = keff_df.loc[water_density,'keff unc']
        dk2 = keff_df.loc[water_density[-1],'keff unc']
        k2_minus_k1 = k2-k1
        k2_times_k1 = k2*k1
        d_k2_minus_k1 = np.sqrt(dk2**2+dk1**2)
        d_k2_times_k1 = k2*k1*np.sqrt((dk2/k2)**2+(dk1/k1)**2)
        rho = (k2-k1)/(k2*k1)*100

        rho_df.loc[water_density,'keff'] = rho
        if k2_minus_k1 != 0:
            d_rho = rho*np.sqrt((d_k2_minus_k1/k2_minus_k1)**2+(d_k2_times_k1/k2_times_k1)**2)
            rho_df.loc[water_density,'keff unc'] = d_rho
        else: rho_df.loc[water_density,'keff unc'] = 0

    print(f"\nDataframe of rho values and their uncertainties:\n{rho_df}\n")
    rho_df.to_csv(f"{rho_csv_name}")


'''
Calculates a few other rod parameters.

rho_csv_name: str, name of CSV of rho values to read from, e.g. "rho.csv"
params_csv_name: str, desired name of CSV of rod parameters, e.g. "rod_parameters.csv"

Does not return anything. Only performs file creation.
'''


def calc_params_void(rho_csv_name, params_csv_name):
    rho_df = pd.read_csv(rho_csv_name, index_col=0)
    rods = [c for c in rho_df.columns.values.tolist() if "unc" not in c]
    heights = rho_df.index.values.tolist()

    beta_eff = 0.0075
    react_add_rate_limit = 0.16
    motor_speed = {"safe": 19, "shim": 11, "reg": 24}  # inches/min

    parameters = ["worth ($)", "max worth added per % height ($/%)", "max worth added per height ($/in)",
                  "reactivity addition rate ($/sec)", "max motor speed (in/min)"]

    # Setup a dataframe to collect rho values
    params_df = pd.DataFrame(columns=parameters)  # use lower cases to match 'rods' def above
    params_df["rod"] = rods
    params_df.set_index("rod", inplace=True)

    for rod in rods:  # We want to sort our curves by rods
        rho = rho_df[f"{rod}"].tolist()
        # worth ($) = rho / beta_eff, rho values are in % rho per NIST standard
        worth = 0.01 * float(max(rho)) / float(beta_eff)
        params_df.loc[rod, parameters[0]] = worth

        int_eq = np.polyfit(heights, rho, 3)  # coefs of integral worth curve equation
        dif_eq = -1 * np.polyder(int_eq)
        max_worth_rate_per = 0.01 * max(np.polyval(dif_eq, heights)) / float(beta_eff)
        params_df.loc[rod, parameters[1]] = max_worth_rate_per

        max_worth_rate_inch = float(max_worth_rate_per) / float(CM_PER_PERCENT_HEIGHT) * 2.54
        params_df.loc[rod, parameters[2]] = max_worth_rate_inch

        # Normal rod motion speed is about:
        # 19 inches (48.3 cm) per minute for the Safe rod,
        # 11 inches (27.9 cm) per minute for the Shim rod,
        # 24 inches (61.0 cm) per minute for the Reg rod.

        react_add_rate = motor_speed[rod] * max_worth_rate_inch / 60
        params_df.loc[rod, parameters[3]] = react_add_rate

        max_motor_speed = 1 / max_worth_rate_inch * react_add_rate_limit * 60
        params_df.loc[rod, parameters[4]] = max_motor_speed

    print(f"\nVarious rod parameters:\n{params_df}")
    params_df.to_csv(params_csv_name)

if __name__ == '__main__':
    main()