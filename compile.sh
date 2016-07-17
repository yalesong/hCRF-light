# We use the LBFGS library as an off-the-shelf optimizer. Compile it first:
cd ./lib/liblbfgs
./configure
make
cd ../..
# Next, compile the hCRF library. 
make
make distribute
