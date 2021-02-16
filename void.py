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
from scipy import stats
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from mcnp_funcs import *

FILEPATH = os.path.dirname(os.path.abspath(__file__))
WATER_MAT_CARD = '102'
WATER_DENSITIES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0] # np.arange(start=0.1,stop=1.0,step=0.1)
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
    for rho_or_dollars in ['rho','dollars']: plot_data_void(KEFF_CSV_NAME, RHO_CSV_NAME, FIGURE_NAME, rho_or_dollars)

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
    rho_df.columns = ['rho','rho unc']
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
        k2 = keff_df.loc[1.0,'keff']
        dk1 = keff_df.loc[water_density,'keff unc']
        dk2 = keff_df.loc[1.0,'keff unc']
        k2_minus_k1 = k2-k1
        k2_times_k1 = k2*k1
        d_k2_minus_k1 = np.sqrt(dk2**2+dk1**2)
        d_k2_times_k1 = k2*k1*np.sqrt((dk2/k2)**2+(dk1/k1)**2)
        rho = -(k2-k1)/(k2*k1)*100

        rho_df.loc[water_density,'rho'] = rho
        if k2_minus_k1 != 0:
            d_rho = rho*np.sqrt((d_k2_minus_k1/k2_minus_k1)**2+(d_k2_times_k1/k2_times_k1)**2)
            rho_df.loc[water_density,'rho unc'] = d_rho
        else: rho_df.loc[water_density,'rho unc'] = 0

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


'''
Plots integral and differential worths given a CSV of rho and uncertainties.

rho_csv_name: str, name of CSV of rho and uncertainties, e.g. "rho.csv"
figure_name: str, desired name of resulting figure, e.g. "figure.png"

Does not return anything. Only produces a figure.

NB: Major plot settings have been organized into variables for your personal convenience.
'''
def plot_data_void(keff_csv_name, rho_csv_name, figure_name, rho_or_dollars):
    if rho_or_dollars.lower() in ['r','p','rho']: rho_or_dollars = 'rho'
    elif rho_or_dollars.lower() in ['d','dollar','dollars']: rho_or_dollars = 'dollars'

    keff_df = pd.read_csv(keff_csv_name, index_col=0)
    rho_df = pd.read_csv(rho_csv_name, index_col=0)
    water_densities = rho_df.index.values.tolist()

    # Personal parameters, to be used in plot settings below.
    label_fontsize = 16
    legend_fontsize = "x-large"
    # fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    my_dpi = 320
    x_label = "Water density (g/cc)"
    y_label_keff = r"Effective multiplication factor ($k_{eff}$)"
    if rho_or_dollars == 'rho': y_label_void = r"Void coefficient (\$/%)"
    elif rho_or_dollars == 'dollars': y_label_void = r"Void coefficient ((%$\Delta$k/k)/%)"
    plot_color = ["tab:red","tab:blue","tab:green"]

    ax_keff_x_min, ax_keff_x_max = 0, 2
    ax_keff_y_min, ax_keff_y_max = 0.85, 1.2
    ax_keff_x_major_ticks_interval = 0.1
    ax_keff_x_minor_ticks_interval = 0.05
    ax_keff_y_major_ticks_interval = 0.05
    ax_keff_y_minor_ticks_interval = 0.025

    ax_rho_x_min, ax_rho_x_max = 0, 2
    ax_rho_y_min, ax_rho_y_max = -15.5, 0.5
    ax_rho_y_min_dollars, ax_rho_y_max_dollars = -15.5, 0.5
    ax_rho_x_major_ticks_interval, ax_rho_x_minor_ticks_interval = 0.1, 0.05
    ax_rho_y_major_ticks_interval, ax_rho_y_minor_ticks_interval = 1, 0.5
    ax_rho_y_major_ticks_interval_dollars, ax_rho_y_minor_ticks_interval_dollars = 1, 0.5

    ax_void_x_min, ax_void_x_max = 0, 2
    ax_void_y_min, ax_void_y_max = 0.85, 1.2
    ax_void_y_min_dollars, ax_void_y_max_dollars = 0.85, 1.2
    ax_void_x_major_ticks_interval = 0.1
    ax_void_x_minor_ticks_interval = 0.05
    ax_void_y_major_ticks_interval = 0.05
    ax_void_y_minor_ticks_interval = 0.025
    ax_void_y_major_ticks_interval_dollars = 1
    ax_void_y_minor_ticks_interval_dollars = 0.5

    fig, axs = plt.subplots(2, 1, figsize=(1636 / 96, 3 * 673 / 96), dpi=my_dpi, facecolor='w', edgecolor='k')
    ax_keff, ax_rho, ax_void = axs[0], axs[1], axs[2]  # integral, differential worth on top, bottom, resp.


    # Plot data for keff.
    y_keff = keff_df[f'keff'].tolist()
    y_keff_unc = keff_df[f'keff unc'].tolist()

    ax_keff.errorbar(water_densities, y_keff, yerr=y_keff_unc,
                     marker="o", ls="none",
                     color=plot_color[0], elinewidth=2, capsize=3, capthick=2)

    eq_keff = np.polyfit(water_densities, y_keff, 3)  # coefs of integral worth curve equation
    x_fit = np.linspace(water_densities[0], water_densities[-1], len(water_densities))
    y_fit_keff = np.polyval(eq_keff, x_fit)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit_keff)
    ax_keff.plot(x_fit, y_fit_keff, color=plot_color[0], label=r'y={:.2f}x+{:.2f}   $R^2$={:.2f}    $\sigma$={:.2f}'.format(slope,intercept, r_value, std_err))

    # Plot data for reactivity
    y_rho = rho_df[f'rho'].tolist()
    y_rho_unc = rho_df[f'rho unc'].tolist()

    ax_rho.errorbar(water_densities, y_rho, yerr=y_rho_unc,
                     marker="o", ls="none",
                     color=plot_color[1], elinewidth=2, capsize=3, capthick=2)

    eq_rho = np.polyfit(water_densities, y_keff, 3)  # coefs of integral worth curve equation
    x_fit = np.linspace(water_densities[0], water_densities[-1], len(water_densities))
    y_fit_rho = np.polyval(eq_rho, x_fit)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit_rho)
    ax_keff.plot(x_fit, y_fit_rho, color=plot_color[1], label=r'y={:.2f}x+{:.2f}   $R^2$={:.2f}    $\sigma$={:.2f}'.format(slope,intercept, r_value, std_err))


    # Keff plot settings
    ax_keff.set_xlim([ax_keff_x_min, ax_keff_x_max])
    ax_keff.set_ylim([ax_keff_y_min, ax_keff_y_max])

    ax_keff.xaxis.set_major_locator(MultipleLocator(ax_keff_x_major_ticks_interval))
    ax_keff.yaxis.set_major_locator(MultipleLocator(ax_keff_y_major_ticks_interval))

    ax_keff.minorticks_on()
    ax_keff.xaxis.set_minor_locator(MultipleLocator(ax_keff_x_minor_ticks_interval))
    ax_keff.yaxis.set_minor_locator(MultipleLocator(ax_keff_y_minor_ticks_interval))

    ax_keff.tick_params(axis='both', which='major', labelsize=label_fontsize)

    ax_keff.grid(b=True, which='major', color='#999999', linestyle='-', linewidth='1')
    ax_keff.grid(which='minor', linestyle=':', linewidth='1', color='gray')

    ax_keff.set_xlabel(x_label, fontsize=label_fontsize)
    ax_keff.set_ylabel(y_label_keff, fontsize=label_fontsize)
    ax_keff.legend(title=f'Key', title_fontsize=legend_fontsize, ncol=1, fontsize=legend_fontsize, loc='lower right')

    # Reactivity worth plot settings
    ax_rho.set_xlim([ax_rho_x_min, ax_rho_x_max])
    ax_rho.set_ylim([ax_rho_y_min, ax_rho_y_max])

    ax_rho.minorticks_on()

    ax_rho.xaxis.set_major_locator(MultipleLocator(ax_rho_x_major_ticks_interval))
    ax_rho.yaxis.set_major_locator(MultipleLocator(ax_rho_y_major_ticks_interval))

    ax_rho.xaxis.set_minor_locator(MultipleLocator(ax_rho_x_minor_ticks_interval))
    ax_rho.yaxis.set_minor_locator(MultipleLocator(ax_rho_y_minor_ticks_interval))

    # Overwrite set_ylim above for dollar units
    if rho_or_dollars == "dollars":
        ax_rho.set_ylim([-0.25, 3.5])  # Use for dollars units
        ax_rho.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Use for 2 decimal places after 0. for dollars units

        ax_rho.xaxis.set_major_locator(MultipleLocator(10))
        ax_rho.yaxis.set_major_locator(MultipleLocator(0.5))

        ax_rho.xaxis.set_minor_locator(MultipleLocator(2.5))
        ax_rho.yaxis.set_minor_locator(MultipleLocator(0.125))


    ax_rho.tick_params(axis='both', which='major', labelsize=label_fontsize)

    ax_rho.grid(b=True, which='major', color='#999999', linestyle='-', linewidth='1')
    ax_rho.grid(which='minor', linestyle=':', linewidth='1', color='gray')

    ax_rho.set_xlabel(x_label, fontsize=label_fontsize)
    ax_rho.set_ylabel(y_label_rho, fontsize=label_fontsize)
    ax_rho.legend(title=f'Key', title_fontsize=legend_fontsize, ncol=4, fontsize=legend_fontsize, loc='upper right')


    plt.savefig(f"{figure_name.split('.')[0]}_{rho_or_dollars}.{figure_name.split('.')[-1]}", bbox_inches='tight',
                pad_inches=0.1, dpi=my_dpi)
    print(
        f"\nFigure '{figure_name.split('.')[0]}_{rho_or_dollars}.{figure_name.split('.')[-1]}' saved!\n")  # no space near \


if __name__ == '__main__':
    main()