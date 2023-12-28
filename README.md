# CesiumTest

### Requirements

Installed CUDA. I use version CUDA 12.3 installed on Ubuntu 23.10

Input file `vancouver.data` should be placed in `data` subdirectory of executable path

### Building
```
nvcc main.cu -o CesiumTest
```
### Running
```
./CesiumTest
```
All mipmap output files will be written to `data` subdirectory
