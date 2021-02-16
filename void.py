"""
VOID COEFFICIENT (MCNP)

Written by Patrick Park (RO, Physics '22)
ppark@reed.edu

This project should be available at
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
from scipy.optimize import curve_fit
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

    # if not check_run_mcnp(): sys.exit()

    # Run MCNP for all .i files in f".\{inputs_folder_name}".
    tasks = 12 # get_tasks()
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
    calc_params_void(RHO_CSV_NAME, PARAMS_CSV_NAME)
    for rho_or_dollars in ['rho','dollars']: plot_data_void(KEFF_CSV_NAME, RHO_CSV_NAME, PARAMS_CSV_NAME, FIGURE_NAME, rho_or_dollars)

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
    rho_df.columns = ['rho', 'rho unc']
    rho_df["density"] = water_densities
    rho_df.set_index("density", inplace=True)

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
        dollars = 0.01*rho/BETA_EFF

        rho_df.loc[water_density, 'rho'] = rho
        rho_df.loc[water_density, 'dollars'] = dollars
        # while the 'dollars' (and 'dollars unc') columns are not in the original rho_df definition,
        # simply defining a value inside it automatically adds the column
        if k2_minus_k1 != 0:
            rho_unc = rho*np.sqrt((d_k2_minus_k1/k2_minus_k1)**2+(d_k2_times_k1/k2_times_k1)**2)
            dollars_unc = rho_unc/100/BETA_EFF
            rho_df.loc[water_density,'rho unc'], rho_df.loc[water_density,'dollars unc'] = rho_unc, dollars_unc
        else: rho_df.loc[water_density, 'rho unc'], rho_df.loc[water_density, 'dollars unc'] = 0, 0

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

    parameters = ['density', 'D density %', 'D rho', 'rho unc', 'D dollars', 'dollars unc',
                  'void rho', 'void rho avg', 'void rho unc', 'void dollars', 'void dollars avg', 'void dollars unc']

    # Setup a dataframe to collect rho values
    # Here, 'D' stands for $\Delta$, i.e., macroscopic change
    params_df = pd.DataFrame(columns=parameters)  # use lower cases to match 'rods' def above
    params_df['density'] = WATER_DENSITIES
    params_df.set_index('density', inplace=True)
    # params_df['D density %'] = [100*round(1.0-x, 1) for x in WATER_DENSITIES]
    params_df['D rho'] = rho_df['rho']
    params_df['rho unc'] = rho_df['rho unc']
    params_df['D dollars'] = rho_df['dollars']
    params_df['dollars unc'] = rho_df['dollars unc']

    for water_density in WATER_DENSITIES:
        if water_density == 1.0:
            params_df.loc[water_density, 'D density %'] = 100 * round(1.0 - water_density, 1)
            params_df.loc[water_density, 'void rho'], params_df.loc[water_density, 'void rho unc'], \
            params_df.loc[water_density, 'void dollars'], params_df.loc[water_density, 'void dollars unc'], \
            params_df.loc[water_density, 'void rho avg'], params_df.loc[water_density, 'void dollars avg'] = 0, 0, 0, 0, 0, 0
        else:
            params_df.loc[water_density, 'D density %'] = 100 * round(1.0 - water_density, 1)
            params_df.loc[water_density, 'void rho'] = params_df.loc[water_density, 'D rho'] / params_df.loc[water_density, 'D density %']
            """params_df.loc[water_density, 'void rho avg'] =np.mean(np.polyval(np.polyfit(
                [x for x in params_df['D density %'].tolist() if str(x).lower() != 'nan'],
                [y for y in params_df['void rho'].tolist() if str(y).lower() != 'nan'], 1),
                [x for x in params_df['D density %'].tolist() if str(x).lower() != 'nan']))"""
            params_df.loc[water_density, 'void rho unc'] = params_df.loc[water_density, 'rho unc'] / params_df.loc[water_density, 'D density %']
            params_df.loc[water_density, 'void dollars'] = params_df.loc[water_density, 'D dollars'] / params_df.loc[water_density, 'D density %']
            """params_df.loc[water_density, 'void dollars avg'] = np.mean(np.polyval(np.polyfit(
                [x for x in params_df['D density %'].tolist() if str(x).lower() != 'nan'],
                [y for y in params_df['void dollars'].tolist() if str(y).lower() != 'nan'], 1),
                [x for x in params_df['D density %'].tolist() if str(x).lower() != 'nan']))"""
            params_df.loc[water_density, 'void dollars unc'] = params_df.loc[water_density, 'dollars unc'] / params_df.loc[water_density, 'D density %']

    for water_density in WATER_DENSITIES:
        x = [i for i in WATER_DENSITIES if water_density<= i <= 1.0]
        if len(x) > 1:
            y_rho = params_df.loc[x, 'void rho'].tolist()
            params_df.loc[water_density, 'void rho avg'] = np.mean(np.polyval(np.polyfit(x, y_rho, 1), x))
            y_dollars = params_df.loc[x, 'void dollars'].tolist()
            params_df.loc[water_density, 'void dollars avg'] = np.mean(np.polyval(np.polyfit(x, y_dollars, 1), x))

    print(f"\nVarious rod parameters:\n{params_df}")
    params_df.to_csv(params_csv_name)

"""
prints dictionary of 'polynomial' and 'r-squared'
e.g., {'polynomial': [-0.0894, 0.234, 0.8843], 'r-squared': 0.960}
"""
def find_poly_reg(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['r-squared'] = ssreg / sstot
    return results

'''
Plots integral and differential worths given a CSV of rho and uncertainties.

rho_csv_name: str, name of CSV of rho and uncertainties, e.g. "rho.csv"
figure_name: str, desired name of resulting figure, e.g. "figure.png"

Does not return anything. Only produces a figure.

NB: Major plot settings have been organized into variables for your personal convenience.
'''
def plot_data_void(keff_csv_name, rho_csv_name, params_csv_name, figure_name, rho_or_dollars, for_fun=False):
    if rho_or_dollars.lower() in ['r','p','rho']: rho_or_dollars = 'rho'
    elif rho_or_dollars.lower() in ['d','dollar','dollars']: rho_or_dollars = 'dollars'

    keff_df = pd.read_csv(keff_csv_name, index_col=0)
    rho_df = pd.read_csv(rho_csv_name, index_col=0)
    params_df = pd.read_csv(params_csv_name, index_col=0)
    water_densities = rho_df.index.values.tolist()

    # Personal parameters, to be used in plot settings below.
    label_fontsize = 16
    legend_fontsize = "x-large"
    # fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    my_dpi = 320
    x_label = r"Water density "# (g/cm$^3$)"
    y_label_keff, y_label_rho, y_label_void = r"Effective multiplication factor ($k_{eff}$)", \
                                              r"Reactivity ($\%\Delta k/k$)", \
                                              r"Void coefficient ((%$\Delta k/k$)/%)"
    if rho_or_dollars == 'dollars':
        y_label_rho, y_label_void= r"Reactivity ($\Delta$\$)", r"Void coefficient (\$/%)"

    plot_color = ["tab:red","tab:blue","tab:green"]

    ax_x_min, ax_x_max = 0.05, 1.05
    ax_x_major_ticks_interval, ax_x_minor_ticks_interval = 0.1, 0.025
    if for_fun:
        ax_x_min, ax_x_max = 0, 2
        ax_x_major_ticks_interval, ax_x_minor_ticks_interva = 0.1, 0.05

    ax_keff_y_min, ax_keff_y_max = 0.8, 1.15
    ax_keff_y_major_ticks_interval, ax_keff_y_minor_ticks_interval = 0.05, 0.025

    ax_rho_y_min, ax_rho_y_max = -16, 1
    ax_rho_y_major_ticks_interval, ax_rho_y_minor_ticks_interval = 2, 1
    if rho_or_dollars == 'dollars':
        ax_rho_y_min, ax_rho_y_max = -21, 1.0
        ax_rho_y_major_ticks_interval, ax_rho_y_minor_ticks_interval = 2, 1

    ax_void_y_min, ax_void_y_max = -0.4, 0.1
    ax_void_y_major_ticks_interval, ax_void_y_minor_ticks_interval = 0.1, 0.025
    if rho_or_dollars == 'dollars':
        ax_void_y_min, ax_void_y_max = -0.5, 0.1
        ax_void_y_major_ticks_interval, ax_void_y_minor_ticks_interval = 0.1, 0.025

    fig, axs = plt.subplots(3, 1, figsize=(1636 / 96, 3 * 673 / 96), dpi=my_dpi, facecolor='w', edgecolor='k')
    ax_keff, ax_rho, ax_void = axs[0], axs[1], axs[2]  # integral, differential worth on top, bottom, resp.

    # Plot data for keff.
    x = [water_density for water_density in water_densities if water_density <= 1]
    if for_fun: x = water_densities
    x_fit = np.linspace(min(x), max(x), len(water_densities))
    y_keff, y_keff_unc = [], []
    for water_density in x:
        y_keff.append(keff_df.loc[water_density,'keff']), y_keff_unc.append(keff_df.loc[water_density,'keff unc'])

    ax_keff.errorbar(x, y_keff, yerr=y_keff_unc,
                     marker="o", ls="none",
                     color=plot_color[0], elinewidth=2, capsize=3, capthick=2)

    eq_keff = find_poly_reg(x, y_keff, 2)['polynomial']  # n=2 order fit
    r2_keff = find_poly_reg(x, y_keff, 2)['r-squared']
    sd_keff = np.average(np.abs(np.polyval(np.polyfit(x, y_keff, 2), x) - y_keff))
    y_fit_keff = np.polyval(eq_keff, x)

    ax_keff.plot(x, y_fit_keff, color=plot_color[0],
                 label=r'y=-{:.3f}$x^2$+{:.2f}$x$+{:.2f},  $R^2$={:.2f},  $\sigma$={:.4f}'.format(
                     np.abs(eq_keff[0]),eq_keff[1], eq_keff[2], r2_keff, sd_keff))

    # Plot data for reactivity
    y_rho, y_rho_unc = [], []
    for water_density in x:
        if rho_or_dollars == 'rho': y_rho.append(rho_df.loc[water_density,'rho']), y_rho_unc.append(rho_df.loc[water_density,'rho unc'])
        if rho_or_dollars == 'dollars': y_rho.append(rho_df.loc[water_density, 'dollars']), y_rho_unc.append(rho_df.loc[water_density, 'dollars unc'])

    ax_rho.errorbar(x, y_rho, yerr=y_rho_unc,
                     marker="o", ls="none",
                     color=plot_color[1], elinewidth=2, capsize=3, capthick=2)

    eq_rho = find_poly_reg(x, y_rho, 2)['polynomial']  # n=2 order fit
    r2_rho = find_poly_reg(x, y_rho, 2)['r-squared']
    sd_rho = np.average(np.abs(np.polyval(np.polyfit(x, y_rho, 2), x) - y_rho))
    y_fit_rho = np.polyval(eq_rho, x_fit)

    ax_rho.plot(x_fit, y_fit_rho, color=plot_color[1],
                label=r'y=-{:.1f}$x^2$+{:.0f}$x${:.0f},  $R^2$={:.2f},  $\sigma$={:.2f}'.format(
                    np.abs(eq_rho[0]), eq_rho[1], eq_rho[2], r2_rho, sd_rho))

    # Plot data for void
    y_void, y_void_unc = [], []
    for water_density in x:
        if rho_or_dollars == 'rho': y_void.append(params_df.loc[water_density,'void rho']), y_void_unc.append(params_df.loc[water_density, 'void rho unc'])
        else: y_void.append(params_df.loc[water_density, 'void dollars']), y_void_unc.append(params_df.loc[water_density, 'void dollars unc'])

    ax_void.errorbar(x, y_void, yerr=y_void_unc,
                     marker="o", ls="none",
                     color=plot_color[2], elinewidth=2, capsize=3, capthick=2)

    eq_void = find_poly_reg(x, y_void, 1)['polynomial']
    r2_void = find_poly_reg(x, y_void, 1)['r-squared']
    sd_void = np.average(np.abs(np.polyval(np.polyfit(x, y_void, 1), x) - y_void))
    y_fit_void = np.polyval(eq_void, x_fit)

    ax_void.plot(x_fit, y_fit_void, color=plot_color[2],
                label=r'y={:.2f}$x${:.2f},  $R^2$={:.2f},  $\bar x$$\pm\sigma$={:.3f}$\pm${:.3f}'.format(
                    np.abs(eq_void[0]), eq_void[1], r2_void, np.mean(y_fit_void), sd_void))

    eq_void_der = -1*np.polyder(eq_rho)/100  # n=2 order fit
    y_fit_void_der = np.polyval(eq_void_der, x_fit)

    ax_void.plot(x_fit, y_fit_void_der, color=plot_color[2], linestyle='dashed',
                label=r'y={:.2f}$x${:.2f},  $\bar x$={:.3f}'.format(
                    np.abs(eq_void_der[0]), eq_void_der[1], np.mean(y_fit_void_der)))




    # Keff plot settings
    ax_keff.set_xlim([ax_x_min, ax_x_max])
    ax_keff.set_ylim([ax_keff_y_min, ax_keff_y_max])
    ax_keff.xaxis.set_major_locator(MultipleLocator(ax_x_major_ticks_interval))
    ax_keff.yaxis.set_major_locator(MultipleLocator(ax_keff_y_major_ticks_interval))
    ax_keff.minorticks_on()
    ax_keff.xaxis.set_minor_locator(MultipleLocator(ax_x_minor_ticks_interval))
    ax_keff.yaxis.set_minor_locator(MultipleLocator(ax_keff_y_minor_ticks_interval))

    ax_keff.tick_params(axis='both', which='major', labelsize=label_fontsize)
    ax_keff.grid(b=True, which='major', color='#999999', linestyle='-', linewidth='1')
    ax_keff.grid(which='minor', linestyle=':', linewidth='1', color='gray')

    ax_keff.set_xlabel(x_label, fontsize=label_fontsize)
    ax_keff.set_ylabel(y_label_keff, fontsize=label_fontsize)
    ax_keff.legend(title=f'Key', title_fontsize=legend_fontsize, ncol=1, fontsize=legend_fontsize, loc='lower right')


    # Reactivity worth plot settings
    ax_rho.set_xlim([ax_x_min, ax_x_max])
    ax_rho.set_ylim([ax_rho_y_min, ax_rho_y_max])
    ax_rho.xaxis.set_major_locator(MultipleLocator(ax_x_major_ticks_interval))
    ax_rho.yaxis.set_major_locator(MultipleLocator(ax_rho_y_major_ticks_interval))
    ax_rho.minorticks_on()
    ax_rho.xaxis.set_minor_locator(MultipleLocator(ax_x_minor_ticks_interval))
    ax_rho.yaxis.set_minor_locator(MultipleLocator(ax_rho_y_minor_ticks_interval))

    # Use for 2 decimal places after 0. for dollars units
    if rho_or_dollars == "dollars": ax_rho.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax_rho.tick_params(axis='both', which='major', labelsize=label_fontsize)
    ax_rho.grid(b=True, which='major', color='#999999', linestyle='-', linewidth='1')
    ax_rho.grid(which='minor', linestyle=':', linewidth='1', color='gray')

    ax_rho.set_xlabel(x_label, fontsize=label_fontsize)
    ax_rho.set_ylabel(y_label_rho, fontsize=label_fontsize)
    ax_rho.legend(title=f'Key', title_fontsize=legend_fontsize, ncol=1, fontsize=legend_fontsize, loc='lower right')


    # Void worth plot settings
    ax_void.set_xlim([ax_x_min, ax_x_max])
    ax_void.set_ylim([ax_void_y_min, ax_void_y_max])
    ax_void.xaxis.set_major_locator(MultipleLocator(ax_x_major_ticks_interval))
    ax_void.yaxis.set_major_locator(MultipleLocator(ax_void_y_major_ticks_interval))
    ax_void.minorticks_on()
    ax_void.xaxis.set_minor_locator(MultipleLocator(ax_x_minor_ticks_interval))
    ax_void.yaxis.set_minor_locator(MultipleLocator(ax_void_y_minor_ticks_interval))

    # Use for 2 decimal places after 0. for dollars units
    if rho_or_dollars == "dollars": ax_void.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax_void.tick_params(axis='both', which='major', labelsize=label_fontsize)
    ax_void.grid(b=True, which='major', color='#999999', linestyle='-', linewidth='1')
    ax_void.grid(which='minor', linestyle=':', linewidth='1', color='gray')

    ax_void.set_xlabel(x_label, fontsize=label_fontsize)
    ax_void.set_ylabel(y_label_void, fontsize=label_fontsize)
    ax_void.legend(title=f'Key', title_fontsize=legend_fontsize, ncol=1, fontsize=legend_fontsize, loc='lower right')


    plt.savefig(f"{figure_name.split('.')[0]}_{rho_or_dollars}.{figure_name.split('.')[-1]}", bbox_inches='tight',
                pad_inches=0.1, dpi=my_dpi)
    print(
        f"\nFigure '{figure_name.split('.')[0]}_{rho_or_dollars}.{figure_name.split('.')[-1]}' saved!\n")  # no space near \


if __name__ == '__main__':
    main()