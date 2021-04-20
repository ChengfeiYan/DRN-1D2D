# DRN-1D2D
protein contact map prediction


## Requirements
- python3.6
  1. pytorch1.6  
  2. numpy
  3. matplotlib
  4. pickle
- other packages
  1. [alnstats](https://github.com/psipred/metapsicov/tree/master/src)
  2. [fasta2aln](https://github.com/kad-ecoli/hhsuite2/blob/master/bin/fasta2aln)
  3. [Loadhmm](https://github.com/j3xugit/RaptorX-Contact/blob/master/Common/LoadHHM.py)
  4. [CCMpred](https://github.com/soedinglab/CCMpred)
  5. [TGT_Package](https://github.com/realbigws/TGT_Package)
  6. [Predict_Property](https://github.com/realbigws/Predict_Property)
  7. [hh-suite](https://github.com/soedinglab/hh-suite)

## Installation
### 1. Install python environments(pytorch):
#### Linux and Windows(Conda)
    # CUDA 9.2
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=9.2 -c pytorch

    # CUDA 10.1
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

    # CUDA 10.2
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

    # CPU Only
    conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch
#### Linux and Windows(Wheel)
    # CUDA 10.2
    pip install torch==1.6.0 torchvision==0.7.0

    # CUDA 10.1
    pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

    # CUDA 9.2
    pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

    # CPU only
    pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    
## 2. Install other packages

### 2.1. install DRN-1D2D
    git clone https://github.com/ChengfeiYan/DRN-1D2D.git

### 2.2. install alnstats; fasta2aln
    cd ./DRN-1D2D/bin
    gcc alnstats.c -lm -o alnstats
    g++ fasta2aln.cpp -o fasta2aln

### 2.3. install CCMpred
    git clone --recursive https://github.com/soedinglab/CCMpred.git
    cd CCMpred
    cmake -DWITH_CUDA=OFF
    make
    cd ../

### 2.4. install Predict_Property
    git clone https://github.com/realbigws/Predict_Property
    cd Predict_Property
    cd source_code/
    make
    cd ../../

### 2.5. install TGT_Package
    git clone https://github.com/realbigws/TGT_Package
    cd TGT_Package
    cd source_code/
    make
    cd ../../

### 2.6. install hh-suite
    git clone https://github.com/soedinglab/hh-suite.git
    mkdir -p hh-suite/build && cd hh-suite/build
    cmake -DCMAKE_INSTALL_PREFIX=. ..
    make -j 4 && make install
    export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH"

## Usage:
    python predictor.py a3m_file fasta_file save_path ncpu  
test:

    cd DRN-1D2D  
    python ./scripts/predictor.py ./example/test.a3m ./example/test.fasta ./example 1
