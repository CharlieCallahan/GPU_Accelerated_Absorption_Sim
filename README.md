# GPU Accelerated Absorption Simulation (GAAS)
A python based rapid and accurate broadband gas spectrum simulator that supports both Voigt and HTP based line shapes. Orders of magnitude faster than HAPI for spectra with more than 1000 absorption transitions. Based on a natively written OpenCL implementation of the complex error function which is derived from [http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package]. 

# Installation
Clone the repo
```bash
git clone https://github.com/CharlieCallahan/GPU_Accelerated_Absorption_Sim.git
```
Install the requirements
```bash
python install.py
```

# Quick Start
The repo comes with an example GUI to get you started. Run it with the following command:
```bash
python gaasGUI.py
```
![Alt Text](./assets/GAAS_GUI.png?raw=true "Title")

# Example
Additionally, the repo includes some example code to demonstrate how to run spectrum simulations using GAAS (example_voigt.py and example_htp.py).

# Paper
Please cite us if you use this code in your research. The paper is currently under review.