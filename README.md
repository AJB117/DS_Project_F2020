# Team 4 Project for CSE 40647

#### Predicting Congressional Party Flips with Binary Classification

Patrick Soga, Connor Delaney, Luke Siela, and Brian Cariddi

---

The purpose of this project is to predict whether the party for a given congressional district's representative will change during an election using the demographic data of that district. Below is a description of the file structure and what each file/folder means.

#### `data`

This contains all project data. `original_feature_data` and `original_label_data` contain the original raw data from [E. Scott Adler's](https://sites.google.com/a/colorado.edu/adler-scott/data/congressional-district-data) personal site and [MIT's Election Data Science Lab](https://history.house.gov/Institution/Election-Statistics/Election-Statistics/), respectively.

`flipped_data` contains all data using whether the congressional district flipped parties or not as a label as well as the feature data concatenated using the 
demographic data collected in `original_feature_data`. 

`scaled_merged_features_and_flipped_labels.csv` is the main cleaned and scaled data file that has all integrated feature and label data. Use this for training/testing models. 

`1978_case_study.csv` contains cleaned and scaled data for the case study for 1978.

#### `data_cleaning_and_scaling`

This contains all scripts for cleaning and scaling the data.  `concat_demographics.py`
combines all demographic feature data
for each congressional district and merges them into a single file. 

`data_reduction_house.py`
removes all unnecessary columns from the election outcome label data such as
candidate name, whether the candidate was a write-in, etc. and then collects
the election outcome label data from all years of interest into a single file.

`get_flip_labels.py` does further processing on the collected election label data
and generates further features such as prev_party and win_ratio. 

`merge_on_id.py`
takes the accumulated feature and label data and joins them based on a custom ID.

`scaling.py` processes that integrated data, standardizing and normalizing features that need to be scaled.

#### `modeling`

This contains all the scripts that run and tune relevant models.
`all_models.py` instantiates various models and trains them. Based on given parameters,
it may oversample or undersample or leave alone the training data for the models.
All results are saved in the `output` folder.

`model_optimization.py` contains a general function used for iterating over a numerical range
representing the numerical value of the hyperparameter of interest and plots
the accuracy/area under the ROC curve against the hyperparameter value.
Change manually to tune different classifiers.

`case_study.py` trains an AdaBoost model using oversampled training data and then
predicts whether the districts in the 1978 case study data flip or do not flip.
It prints the indices of the data objects in the 1978 case study csv for the user to inspect.

`feature_interpretation.py` is very similar to `all_models.py` except it 
performs a hold-out analysis of the features to investigate each feature's impact on accuracy.
All results are saved in the `output` folder and contains each feature paired with the accuracy of the model after removal of that feature.

#### `output`
This contains all model results output by the scripts in `modeling`. Their names do not match what the scripts wills save new files 
as since what is currently in `output` are our group's results which the user can verify. `correlations` contains
the correlation matrix for the features with respect to the flip label.