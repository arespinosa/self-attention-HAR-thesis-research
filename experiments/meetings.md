Meeting 1 (01/22/2026)
- Task 1 — Repo setup & environment 
Navigate to week1_env.md to view the environment setup 
- Task 2 - download and inspect PAMAP2
Currently downloaded the PAMAP2 dataset with command: python dataset_download.py -d pamap2 -z

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
2. Since we were using the pre-trained weights, I ran the command:  TF_CPP_MIN_LOG_LEVEL=3 python main.py --test --dataset pamap2 
3. I created a task3.log and stored the results on there 

My initial analyzation from the classification report is that this HAR model does perform well with the test subjects. The precision for the different 
motion activities are all in the high 90 percentile. Also, the activities like lying, descending stairs, ironing, and vaccum cleaning can be neglected since their support is 0. 
That means that these activities were not done by the test subject. Perhaps, we can try other subjects that do perform these activities to see how they do. 
