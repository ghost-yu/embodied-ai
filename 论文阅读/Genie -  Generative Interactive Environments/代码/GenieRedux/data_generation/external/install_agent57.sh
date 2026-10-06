ENV_NAME="retro_datagen"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)"
cd $SCRIPT_PATH
source ~/.bashrc

git clone https://github.com/michaelnny/deep_rl_zoo.git
cd deep_rl_zoo
git checkout 19e3844ccdd23f8e1c45482daf1bb9020c1bcaad

# Apply patch to add Agent57 support
git apply --index ../agent57_changes.patch

cd $SCRIPT_PATH
if conda env list | grep -q "^$ENV_NAME\s"; then
    conda activate $ENV_NAME
    python agent57_download_checkpoints.py
else
    echo "Conda environment $ENV_NAME not found! Please check your conda setup."
fi