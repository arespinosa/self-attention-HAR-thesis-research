# Project Status

## Current Phase
1. Going to plot the embedding vector extracted from extract_embeddings.py  
2. Going to try and visualize the data with WISDM simialr to the OxWearables dataset 
3. Research how this repo detects the labels. If they're all classified as the same thing or if they're different for each dataset we are dealing with. 
4. Grabbing the embedding vector for that specific input we extracted from random_input.py 

## Completed
1. Environment setup & config information 
2. Download & inspect PAMAP2 dataset 
3. Run pretrained PAMAP2 inference 
4. Created a script that extracted the embeddings plus the activity label it predeicted. 
5. Using the embeddings I was able to extract and creating UMaps to visualize the data. 
6. Modified the UMAP script to display activity name alongside the activity label id. 
7. Utilized the hyper parameter from the OxWearables repo. 
8. Extracted one random data from PAMAP2, nd understand its format characteristics. 

## Next Steps 
1. To be determined. 

## Blockers / Risks
- Unable to connect to the GPU. Although the NVIDIA driver is installed, for some reason the GPU does not get detected in my environment. To finish the remaining tasks of this week, I was able to run with the CPU.
    - nvida-smi 
    - python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU'))"

- Some files were missing in the Git repo because I was getting warning errors when trying to push. 

