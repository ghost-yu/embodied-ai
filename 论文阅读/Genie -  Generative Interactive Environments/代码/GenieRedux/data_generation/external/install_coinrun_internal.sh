# Linux
# apt-get install mpich build-essential qt5-default pkg-config

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)"
cd $SCRIPT_PATH
source ~/.bashrc

# conda env create -f ./coinrun_env.yaml
ENV_NAME=coinrun2
conda create -n $ENV_NAME python=3.6.15 -y
conda activate $ENV_NAME
conda install -c conda-forge opencv -y
conda install -c conda-forge mpi4py==3.1.1 mpich==4.2.2 -y
conda install -c conda-forge pango harfbuzz -y
conda install -c conda-forge libglu libgl glew freeglut -y
conda install -c anaconda qt==5.15.9 -y
# pip install qtpy
# pip install tensorflow==1.12.0  # or tensorflow-gpu
# pip install -r requirements.txt
# git clone https://github.com/openai/coinrun.git
# cd coinrun
# cp $SCRIPT_PATH/random_agent.py $SCRIPT_PATH/coinrun/
# pip install -e .
# PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PATH=$CONDA_PREFIX/bin:$PATH python -c "import coinrun"
