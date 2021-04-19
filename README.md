# DRN-1D2D
protein contact map prediction

## INSTALL

### 1. install DRN-1D2D
`git clone https://github.com/ChengfeiYan/DRN-1D2D.git`

### 2. install alnstats; fasta2aln
`cd bin`  
`gcc alnstats.c -lm -o alnstats`
`g++ fasta2aln.cpp -o fasta2aln`

3.
git clone --recursive https://github.com/soedinglab/CCMpred.git
cd CCMpred
cmake -DWITH_CUDA=OFF
make
cd ../

4.
git clone https://github.com/realbigws/Predict_Property
cd Predict_Property
cd source_code/
	make
cd ../../

5.
git clone https://github.com/realbigws/TGT_Package
cd TGT_Package
cd source_code/
	make
cd ../../

6.
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH"

