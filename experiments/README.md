# Experiments
This folder contains the scripts for generating the data and for the experiments. 

## Data Generation
The file [generate_data.py](generate_data.py) contains the code for the data-generating process. 
It stores the data and the graph in one directory. The directory's path is determined by the paths given in [constants.py](constants.py) 
and the seed used the data-generation. 

## Experiments
The experiments for *small p*, *NOTEARS* and *BoostingDAGs* can be run using the different scripts. Note that you have to install NOTEARS before running the NOTEARS experiments. 
For running the experiments for *CAM*, see [here](../R/run_cam.R).
The evaluation of the different algorithms is also run in R and can be found [here](../R/evaluate.R)
### NOTEARS
For running the simulation studies with the NOTEARS algorithm, you need to install the repository manually. See the [repo](https://github.com/xunzheng/notears) for more details.