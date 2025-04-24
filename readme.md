I recommend use conda to run the program.

Before using, please make sure to install the dependencies.

You can use the command to install all the requirements.

conda create --name Tnpb 
conda activate Tnpb
conda install -c anaconda python=3.6
pip install -r requirements.txt


For prediction:
Run the script predict.py


For Training:
Run the script main.py

You can set all the config as you wish in the script config.py

The default running device is cuda, else will use cpu.


Please remember to put the train set, validation set and test set in the correct path.


All the formation of dataset should be as the same as train set/validation set and test set.


Qingyang Liu
04/15/2025
