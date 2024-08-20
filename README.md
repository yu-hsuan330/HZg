# HZg project

## Set the Conda environment
Run the project by the following environment.
```bash
conda env create -f environment.yml
```

## XGBoost training
First, set up the training configuration in `Configs\TrainConfig*.py`, including the training details and input sample information. 
Then, you can run the project directly with the following command:

```bash
cd XGBoost
python Trainer.py Configs/TrainConfig_file
```
The training model is saved as text file, and the program will automatically store the related plots (e.g., MVA distribution, ROC curve, and variable importance).

### To-do list
- [X] Main training code.
- [ ] Upload the code for predicting the BDT score.
