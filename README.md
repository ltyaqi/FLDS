# FLDS
FLDS: Differentially Private Federated Learning with Double Shufflers

This code primarily uses Python 3.8 and the TensorFlow framework. Below are the server environment configuration instructions, parameter settings, and code execution guide.

Environment Configuration

Operating System Requirement: Windows 10
Processor Requirement: 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz
Install Required Packages: Use pip (Python's package manager) to install all necessary packages. First, ensure that Python and pip are installed, and a virtual environment is created using Anaconda.
Install Required Packages
All required packages can be installed using pip. Install the following packages in the conda virtual environment. It is recommended to use the specified versions, as newer versions might have altered or deprecated certain functions:

absl-py 1.1.0
astor 0.8.1
cached-property 1.5.2
certifi 2020.6.20
cycler 0.11.0
dataclasses 0.8
gast 0.2.2
google-pasta 0.2.0
grpcio 1.46.3
h5py 3.1.0
importlib-metadata 4.8.3
Keras-Applications 1.0.8
Keras-Preprocessing 1.1.2
kiwisolver 1.3.1
Markdown 3.3.7
matplotlib 3.3.4
mpmath 1.2.1
numpy 1.19.4
opt-einsum 3.3.0
Pillow 8.4.0
pip 21.2.2
protobuf 3.19.4
pyparsing 3.0.9
python-dateutil 2.8.2
setuptools 59.6.0
six 1.16.0
sympy 1.1.1
tensorboard 1.15.0
tensorflow-cpu 1.15.0
tensorflow-cpu-estimator 1.15.1
termcolor 1.1.0
tqdm 4.15.0
typing_extensions 4.1.1
Werkzeug 2.0.3
wheel 0.37.1
wincertstore 0.2
wrapt 1.14.1
zipp 3.6.0
Example of installation: pip install numpy==1.19.4

The CPU version of TensorFlow can be installed directly using pip, as shown above. If you want to use the GPU version, ensure that the GPU, Python version, and TensorFlow version correspond correctly.

Parameter Settings

The default parameter configurations are as follows:

Number of clients: Default is 6000
Epoch rounds: Default is 10
Iterations per epoch: Default is 6
Number of clients per iteration: Default is 1000 clients participating in each training iteration
Learning rate: Initially set to 0.1, reduced linearly to 0.05 after 5 epochs, then fixed at 0.05
In the main.py file, the following hyperparameters are primarily adjusted for experiments:

Privacy budget (epsilon): The total privacy budget is set by default to 0.3, with a search range of {0.05, 0.1, 0.15, 0.2, 0.3}. The privacy budget for a single dimension needs to be calculated based on the formula provided in the paper. The document "FLDS Privacy Budget Reference Table.xlsx" provides the relationship between the total privacy budget and the single-dimensional privacy budget.
In the fedbase.py file, the following hyperparameters are mainly adjusted for experiments:

Dimensionality reduction ratio: Default value is 1/50, with a search range of {1/10, 1/50, 1/157}
Adjust the following statement accordingly:
sum_sample_clients = 20
generater_random_matrix(1000, 7850, 20, 157, self.epsilon, self.clip_C)
Specific adjustments need to be made based on the dimensionality reduction ratio.
Code Execution

Different datasets in FLDS have separate code files. The code files are labeled according to the comparison scheme, with the default being the MNIST code. For instance, if running the Fashion-MNIST dataset, you need to copy and replace the MNIST dataset in the FLDS directory.

After installing all the necessary packages and setting the correct hyperparameters, you can use the following command to execute the code. The hyperparameters can be modified using the method below or directly in main.py, while the dimensionality reduction ratio can only be modified in fedbase.py:

Without clipping: python main.py --optimizer="flds1" --epsilon=0.1.545 --learning_rate=0.1 --norm=0.2 --mechanism="laplace"

Comparison without clipping: The code is also in the file "FLDS-MNIST-NoClipping". Replace the above --optimizer="flds1" with --optimizer="dpsgd" or other optimizers available in the subfolder flearn/trainer.

With clipping: python main.py --MAD_n=3.0 --optimizer="clip_flds1" --epsilon=0.1.545 --learning_rate=0.1 --norm=0.2 --mechanism="laplace"

Comparison with clipping: python main.py --optimizer="dpsgd" --epsilon=0.3 --learning_rate=0.1 --mechanism="gaussian"
