# Gathering our thoughts

## Data
Currently:
* shuffled_deduped.parquet serves as the baseline we are all working from (on Josh' server under shared/project)
* The data is split into train, test and val (+ small train), which is just randomly sampled subsets of shuffled_deduped

## Models

### Simple model 
Currently:
* Simple model done, see src/simple_model
* It runs a TF-IDF vectorizer and a truncated SVD before passing it to a logistic regression

### Complex model: 
Currently: 
* (Jakup has a NN)


### To Do:
- [] Part 1: Task 2 (+ 1) finished (visualization and some data exploration)
