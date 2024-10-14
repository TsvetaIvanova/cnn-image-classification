# Neural Network Image Classsification Guide

## Environment Setup

1. Create a new folder and place the assessment files and folders in it.
2. Open a terminal and navigate to the folder that contains the assessment content.
3. Create a virtual environment. - $ python -m venv .venv
4. Activate virtual environment. - $ source .venv/bin/activate
5. Install requirements. - $ pip install -r requirements.txt 

## Verifying the trained model

This section runs the evaluation using the model provided in the assessment zip file. The name of the trained model is ‘final_trained_model.pth’.
When running the evaluation please keep in mind that this model was trained and evaluated using GPUs.
Evaluation can take up to 5 min to complete considering the test dataset is larger than the train and validation datasets.

In the terminal and within the folder that contains the assessment files.

1. Ensure venv is activated. - $ source .venv/bin/activate
2. Navigate to the trained-model folder. - $ cd trained-model
3. Run the offimlo.py - $ python offimlo.py


## Training and evaluating new model


In the terminal and within the folder that contains the assessment files.

1. Ensure venv is activated. - $ source .venv/bin/activate
2. Navigate to the training folder. - $ cd training
3. Run the offimlo-train.py - $ python offimlo-train.py





