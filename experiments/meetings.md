### Meeting 1 (01/22/2026)

- Task 1 — Repo setup & environment
  Navigate to week1_env.md to view the environment setup
- Task 2 - download and inspect PAMAP2
  Currently downloaded the PAMAP2 dataset with command: python dataset_download.py -d pamap2 -z
- Task 3 - Run the pretrained PAMAP2 inference and store results.

## Number of Subjects

- 9 subjects, 1 female & 8 males.
- Subject files: subject101.dat - subject109.dat found in Protocol

## List of activity labels used by repo

It uses 19 distinct labels. The numbers attached are their labeled ids. (Found from readme.pdf)

- 1 lying
- 2 sitting
- 3 standing
- 4 walking
- 5 running
- 6 cycling
- 7 Nordic walking
- 9 watching TV
- 10 computer work
- 11 car driving
- 12 ascending stairs
- 13 descending stairs
- 16 vacuum cleaning
- 17 ironing
- 18 folding laundry
- 19 house cleaning
- 20 playing soccer
- 24 rope jumping
- 0 other (transient activities)

## Wearables Used and Column Representation

There were 4 wearables being used. Heart monitors and 3 IMUs(Inertial Measurement Unit): Hand, Chest,& Ankle.
"Each of the data-files contains 54 columns per row"

- 1 timestamp (s)
- 2 activityID
- 3 heart rate (bpm)
- 4-20 IMU hand
- 21-37 IMU chest
- 38-54 IMU ankl

## Input Tensor Shape

It has 19 sensor channels
It has 33 timesteps

So, input tensor shape is (batch_size, 33, 19). Gathered from configs/data.yaml

Notes/Observations:
The previous research split the subjects into a training, testing, and validation group.
Subject 106 was used for testing, Subject 105 was used for validation, and the rest were used for training.

- Task 3 - Run pretrained PAMAP2 inference

Inside the saved_model.py we can find a fully pretrained model w/ the weights included. This indicates that it is ready for inference.
Steps I took,

1. I created an empty processed folder
2. Since we were using the pre-trained weights, I ran the command: TF_CPP_MIN_LOG_LEVEL=3 python main.py --test --dataset pamap2
3. I created a task3.log and stored the results on there

My initial analyzation from the classification report is that this HAR model does perform well with the test subjects. The precision for the different
motion activities are all in the high 90 percentile. Also, the activities like lying, descending stairs, ironing, and vaccum cleaning can be neglected since their support is 0.
That means that these activities were not done by the test subject. Perhaps, we can try other subjects that do perform these activities to see how they do.

### Meeting 2 (01/29/2026)

Given that we know the input, we want to extract each embedding and the label of that activity. After analyzing the file har_model.py inside the model package, I was able to identify the embedding layer as line 27: x = tf.keras.layers.Dense(n_outputs \* 4, activation='relu')(x). I was able to deduct this since it's the last activation layer before the predictions and it happens after the attention application.

- Created the extract_embeddings.py script
- When ran, using: TF_CPP_MIN_LOG_LEVEL=3 python extract_embeddings.py, it will generate the csv file that should link each embedding to its activity label. It will also include .npy files for embeddings and labels

### Meeting 3 (02/05/2062)

With the extracted embeddings, I referenced another repo: https://github.com/OxWearables/ssl-wearables/blob/main to see how they were able to create UMaps. Gathering from this inspiration, I was able to create UMaps of the raw input and then after training the model. From these visualizations, we are able to identify the patterns of how the activities are clumped up together.

### Meeting 4 (02/18/2026)

From the previous UMaps I generated in Meeting 3, I will be modifying them so that they display the activity labels next to the activity label id for a better analyzation. Secondly, I will be utilizing the hyperparameters from the OxWearables repo (https://github.com/OxWearables/ssl-wearables/blob/main) since they spent some time optimizing it for PAMAP2. Lastly, I will grab a random input from the PAMAP2 dataset and analyze the format characteristics. As we prepare to handle the childrens' dataset, we want to see how this repo is fitting and transforming the data to the model.

### Meeting 5 (02/23/2026)
First thing to note is that after looking at the .csv file for the pamap2_embeddings, we have a total window 83,351 test window samples in total. With this information, we are going to try and match it with the WISDM dataset. 
1. In configs/activity_maps, I created the different activity labels for wisdm. I was able to grab this from make_wisdm.py in the OxFordsWearables 
2. I downloaded the clean data from Venodo website https://zenodo.org/records/6574265#.YovCMi8w1qs 
3. I updated the model.yaml file in configs directory to ensure that the wisdm model would be configured properly 
4. Inside the wisdm_experiemnts folder, I have created a file that allows us to see the input shape of the wisdm data. I created the model, and stored the predictions. 
5. In that same folder, I reused the script to create the different UMAPs for WISDM. 

Apart from the wisdm update, I cleaned up the codebase by creating a folder where the pamap2_experiments are stored as well. I also updated the UMAP Visualizations where the singular input embedding is now on there as well. 

### Meeting (04/09)
We have been working on a Notebook in Colab that has been helping with the data preprocesing for the children's dataset. We split up the large amount of data and focused on one participant sample. From this participant, we are visualizing the data to see if their are missing values that require interpolation. Afterwards, we are resampling the children's data from 90 Hz to 30 Hz. 

### Meeting (04/16) 
With the one participant sample, we are now going to feed that input to this model. We will generate the UMAPs and see how well it performed on this data and classifying the activities. 