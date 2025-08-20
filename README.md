# Timing-analysis
A programe  that allows for the automated timing analysis of X-ray data. This codes was developed to automate the standardized data reduction on the retrieved RXTE-PCA data.

# Contribution
Anyone interested in contributing to this project is very welcome to do so. Whether it's submitting a bug report, requesting an improvement, or simply pointing out something unclear your feedback is valuable and appreciated.
If you'd like to contribute code (fixes, improvements, or additions), please consider opening an issue or reaching out first. Pull requests are welcome.

This project is still a work in progress, and even small contributions can help make it more useful for others working with RXTE-PCA data.

Thank you

# Disclaimer
This Codes is licensed under the MIT License, which means it is free for you to use, modify, and distribute.

This project is not affiliated with the RXTE mission, NASA, the Goddard Space Flight Center (GSFC), or HEASARC. It is an independent effort, and under no circumstances should this project be considered endorsed or recommended by any of the aforementioned agencies or organizations.

The author does not claim originality for the core functionalities implemented in these scripts. They are based on standard tools and procedures provided by HEASARC and other established resources. These scripts simply adapt those tools to streamline workflows for handling large volumes of RXTE-PCA data.

# Prerequisites

- stingray
- astropy


# Tested on

- Fedora Linux 42 
- RXTE-PCA data from HEASARC

# About the files
- zn_spin_append.py - This is a python script which runs to identify the pulse period(by performing Z_n search from Stingray) in the given event file by dividing the event file into a segemts of 512 seconds 
- spin_append.txt - This is a bash script which identifies all the barry center corrected event file in a particular parent directory and ececutes the zn_spin_append.py python script for all the valid event file 
- pulse_profiles.py - This is a python script which takes the best frequency from the .csv file created by zn_spin_append.py script along with the event fiel and creates a pulse profile by folding all the 512 segments and look for the High significance pulses and store the detaisl about that particular segment in a .csv file 
- pulse_profiles.txt - This is a bash script which identifies all the barry center corrected event file in a particular parent directory and ececutes the pulse_profiles.py python script for all the valid event file 
- high_sigma_plot.py - This is a python script which takes .csv file as the input and then indentifies all the pulse profile and save then in a single pdf file
