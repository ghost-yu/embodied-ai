# Linux
# apt-get install mpich build-essential qt5-default pkg-config

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)"
cd $SCRIPT_PATH
source ~/.bashrc

conda env create -f ./coinrun_env.yml
ENV_NAME=coinrun

if conda env list | grep -q "^$ENV_NAME\s"; then
    conda activate $ENV_NAME
    conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    pip install tensorflow==1.12.0
    pip install -r requirements.txt
    git clone https://github.com/openai/coinrun.git
    cd coinrun
    cp $SCRIPT_PATH/random_agent.py $SCRIPT_PATH/coinrun/coinrun/
    pip install -e .
    PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PATH=$CONDA_PREFIX/bin:$PATH python -c "import coinrun"
else
    echo "Conda environment $ENV_NAME not found! Please check your conda setup."
fi
