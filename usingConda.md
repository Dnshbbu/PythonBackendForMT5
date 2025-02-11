# Basic Conda Environment Commands
# List all environments
conda env list

# Create a new environment
conda create --name myenv python=3.8

# Activate an environment
conda activate myenv

# Deactivate current environment
conda deactivate

# Remove an environment
conda env remove --name myenv

# To install packages:
conda install numpy pandas xgboost
conda install pytorch torchvision torchaudio -c pytorch
conda install pyzmq




# Similar to pip freeze: (not recommended)
conda list --export > requirements.txt

# Export full environment: (recommended)
conda env export > environment.yml
conda env create -f environment.yml

# For platform independent environment: This only saves the packages you explicitly installed, not their dependencies, which can be better for cross-platform compatibility.)
conda env export --from-history > environment.yml 


