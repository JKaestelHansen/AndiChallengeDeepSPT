# AndiChallengeDeepSPT

#### Installation
DeepSPT's installation guide utilize conda environment setup, therefore either miniconda or anaconda is required to follow the bellow installation guide.
 - Anaconda install guide: [here](https://www.anaconda.com/download)
 - Mini conda install guide: [here](https://docs.conda.io/en/latest/miniconda.html)

DeepSPT is most easily setup in a new conda environment with dependecies, versions, and channels found in environment_droplet.yml - Open Terminal / Commando prompt at wished location of DeepSPT and run the bash commands below, which creates the environemnt, downloades and installs packages, typically in less than 5 minutes. The code has been tested both on MacOS and Linux operating systems.

```bash
git clone git@github.com:JKaestelHansen/AndiChallengeDeepSPT.git OR git clone https://github.com/JKaestelHansen/AndiChallengeDeepSPT (potentially substitute JKaestelHansen with hatzakislab
cd AndiChallengeDeepSPT
conda env create -f environment_droplet.yml
conda activate DeepSPT
pip install probfit==1.2.0
pip install iminuit==2.11.0

As second option:
git clone git@github.com:JKaestelHansen/AndiChallengeDeepSPT.git OR git clone https://github.com/JKaestelHansen/AndiChallengeDeepSPT (potentially substitute JKaestelHansen with hatzakislab
cd AndiChallengeDeepSPT
conda env create -f environment_droplet_minimal.yml
conda activate simpleDeepSPT
pip install h5py==2.10.0
pip install imagecodecs==2023.3.16
pip install pomegranate==0.14.8
pip install probfit==1.2.0
pip install iminuit==2.11.0

As third option (A big thanks to Konstantinos Tsolakidis for contributing approach):
Especially if running this on an Apple Macbook - M1/M2/M3 processor:

git clone git@github.com:JKaestelHansen/AndiChallengeDeepSPT.git OR git clone https://github.com/JKaestelHansen/AndiChallengeDeepSPT (potentially substitute JKaestelHansen with hatzakislab
cd AndiChallengeDeepSPT

conda create --name simpleDeepSPT
conda activate simpleDeepSPT
conda install pip

brew install HDF5 (install brew and update path, instructions here: "https://brew.sh/")
(if the above command gives you an issue, run "arch -arm64 brew install hdf5")
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/(1.12.0_4 or your version)
OR 
export HDF5_DIR=/opt/homebrew/opt/hdf5 (if hdf5 is installed in the "/opt/homebrew/opt/hdf5" location, you have to check it out first)
pip install --no-binary=h5py h5py

conda env update --file environment_droplet_minimal.yml

pip install csbdeep==0.7.4
pip install cython==0.29.37
conda install imagecodecs==2023.1.23
pip install pomegranate==0.14.9
pip install probfit==1.2.0
pip install iminuit==2.11.0
```

# Files
pred_and_prep_submissionfile.py can be run end-to-end on the path to the data to run predictions on by specifying the path in the script

# Data and models
Please download models and data from https://erda.ku.dk/archives/f4751cdc28fdafdfe429f1cf255564f3/published-archive.html and insert folders in to repo individually
