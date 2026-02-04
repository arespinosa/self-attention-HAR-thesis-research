# Project Status

## Current Phase
1. Loading the pretrained model and preparing a script to retrieve each embedding and the distribution of each label 


## Completed
1. Environment setup & config information 
2. Download & inspect PAMAP2 dataset 
3. Run pretrained PAMAP2 inference 

## Next Steps 
1. Once we create the script, we will then begin to plot it. We can reference the other repo discussed to see how they go about implementing it. 

## Blockers / Risks
- Unable to connect to the GPU. Although the NVIDIA driver is installed, for some reason the GPU does not get detected in my environment. To finish the remaining tasks of this week, I was able to run with the CPU.
    - nvida-smi 
    - python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU'))"

- Some files were missing in the Git repo because I was getting warning errors when trying to push. 

