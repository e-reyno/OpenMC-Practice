from asyncio import ALL_COMPLETED
from inspect import stack
from operator import index
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import diamon_read_data as dia
from scipy.signal import find_peaks

def plot_spect(data):
    energy, flux = extract_spectrum(data)
    plt.xscale("log")
    plt.step(energy, flux, label=data.name)
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Flux (cm$^{-2}$ s$^{-1}$)")
    plt.legend()
    plt.savefig("single_energy_spectrum.png")
    plt.show()


def plot_combined_spect(data_array):
    
    for data in data_array:
        
        energy, flux = extract_spectrum(data)
        plt.xscale("log")
        plt.step(energy, flux, label=data.name)
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Flux (cm$^{-2}$s$^{-1}$)")
        plt.legend(fontsize=12, loc=1)
    plt.savefig("combined_energy_spectra.png")
    plt.show()
    

#def contour_plot()



def plot_detector_counts(rate_data, labels):
    for i, rate in enumerate(rate_data):
        # add cumulative time
        rate['time'] = rate['Dt(s)'].cumsum()
        # plot counts over time
        plt.step(rate["time"], rate["Det1"], label=labels[i])
        plt.xlabel("Time (s)")
        plt.ylabel("Counts")
    plt.savefig("detector_counts.png")
    plt.show()
    
    
def plot_dose_rate(df):
    
    plt.figure(figsize=(16,16))
    ax = df["dose_rate"].plot(kind='bar', yerr=(df["dose_rate_uncert"])/100 * df['dose_rate'], capsize=4, color='purple')
    ax.set_xlabel('Date of Measurements')
    ax.set_ylabel('dose rate ($\mu$ Sv\h)')
    ax.set_xticklabels(df["name"], rotation=20)

    plt.savefig("dose_rate_bar_plot.png")
    plt.show()
    
    
def background_subtraction(data,background):
    
    return data - background


def remove_times(data, time_range):
    
    return 0


def average_daily_data(unfold_data):
    
    fluxes = []
    
    for data in unfold_data:
        
        _, flux = extract_spectrum(data)
        if np.average(flux) > 1e-10:
            fluxes.append(flux)
    avg_flux = np.average(fluxes, axis=0)
    return avg_flux


def neuron_energy_dist():
    return 0


def extract_spectrum(data):
    
    energy = data.energy_bins
    flux = data.flux_bins
    return energy, flux

def get_energy_range(unfold_dataseries):
    
    
    return

def peaks_finder(data):
    
    energy, flux = np.array(extract_spectrum(data))
    #border threshold to reflect change in signal
    #border = np.
    
    flux_peaks_i , flux_peaks = find_peaks(flux, height=0, prominence=0.0001)
    flux_peaks = flux_peaks["peak_heights"]

    energy_peaks = energy[(flux_peaks_i)]
    return flux_peaks, energy_peaks

def get_energy_range(unfold_data):
    
    if isinstance(unfold_data, pd.DataFrame):
        
        thermal = unfold_data.thermal
        epitherm = unfold_data.epi
        fast = unfold_data.fast        
        
        return thermal, epitherm, fast
    
def fit_gaussian_spect():
    
    return 0

def find_abs_error(dataframe):
    
    for i, col in enumerate(dataframe.columns):
        if 'un%' in col:
            dataframe["abs_err " + dataframe.columns[i-1]] = dataframe[dataframe.columns[i-1]] * (dataframe[col]/100)
            
    return dataframe


def stack_bar_plot(data_frame, xlabel=None, ylabel=None):
    #stack_df = (data_frame.filter(cols)).astype(float)

    ax = data_frame[["thermal", "epi", "fast"]].plot(kind='bar', stacked=True,figsize=(20,20))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(data_frame["name"], rotation = 45)
    plt.savefig("energy_ranges_bar_plot.png")
    plt.show()
 
def direction_bar_plot(dataframe):
    
    ax = dataframe[["F", "FL", "FR", "R", "RR", "RL"]].plot(kind='bar',figsize = (12,8))
    ax.set_ylabel("Counts")
    ax.set_xticklabels(dataframe["name"], rotation=60)
    plt.show()

   
def find_spect_peaks(data):
    
    energy_list = []
    flux_list = []
    for spectra in data:
        
        energy, flux = peaks_finder(spectra)
        energy_list.append(energy)
        flux_list.append(flux)
    return energy_list, flux_list

def plot_avg_spect(energy, flux):
    
    plt.step(energy, flux)
    plt.xscale("log")
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Flux (cm$^{-2}$s$^{-1}$)")
    plt.legend()
    plt.savefig("average_energy_spectra.png")
    plt.show()
    
    
folder_path = r"C:\Users\sfs81547\Documents\DIAMON project\TS1 measurements\*"


#reads in the folder contianing data for: unfold, rate and out data separated into columns of the measurement
all_data = dia.read_folder(folder_path)
all_data = np.array(all_data)
all_data = all_data[1:,:]


energy_peaks, flux_peaks = find_spect_peaks(all_data[:,0])
print(energy_peaks)
print(flux_peaks)
avg_flux = average_daily_data(all_data[:,0])
plot_avg_spect(all_data[0,0].energy_bins, avg_flux)

#convert to data series
unfold_datalist = []
for data in all_data[:,0]:
    unfold_data = dia.convert_to_ds(data)
    unfold_datalist.append(unfold_data)

plot_combined_spect(all_data[:,0])
    
unfold_dataframe = pd.DataFrame(unfold_datalist)
direction_bar_plot(unfold_dataframe)
stack_bar_plot(unfold_dataframe)
plot_dose_rate(unfold_dataframe)
names = [data[0].name for data in all_data]
plot_detector_counts(all_data[:,2], names)



