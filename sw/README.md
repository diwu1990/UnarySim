# Overview

This directory contains the required components for cycle-accurate software simulation for unary computing. UnarySim is based on [PyTorch](https://pytorch.org/), a deep learning framework from Facebook, such that the simulation can be done on either CPU or GPU efficiently.

The components included in UnarySim belong to three categories, including 
1. **Bit Stream Manipulation**
2. **Unary Computing Kernel**
3. **Performance Metric**

Among those, components in **Bit Stream Manipulation** and **Unary Computing Kernel** can pysically exist in real hardware, while those from **Performance Metric** is usually for performance analysis.

## Prerequisite
1. PyTorch version >= 1.0
2. Python version >= 3.6
3. [pylfsr](https://github.com/Nikeshbajaj/Linear_Feedback_Shift_Register)

## Data Representation
UnarySim has five categories of data, with each having default data type as _**'torch.float'**_ in PyTorch.

1. **Source Data**: 
The input source data in unary computing need to be ranging from _0_ to _1_ in **unipolar** format, or from _-1_ to _1_ in **bipolar** format. 
The input source data (_source_) is scaled to a certain range (as in _unipolar/bipolar_ format) from the raw data (_raw_) with respect to its maximum.
More specifically, such a relationship is formulated as _source = raw / max(raw)_.

2. **Random Number**: 
The random numbers (_rand_) are to be compared with the source data in order to generate the bit streams. To notice, source data is in _unipolar/bipolar_ format, while random numbers are integers. 
To compare them, source data requires to scale up by the _bitwidth_ of random numbers. 
At each cycle, if _round(source * 2^bitwidth) > rand_, a bit of logic 1 in the bit stream will be generated; otherwise, a bit of logic 0 will be generated.

3. **Bit Stream**: 
At each cycle, the bits in bit streams physically flow through cascaded logic kernels, and they count for most of the memory space during simulation.

4. **Bit Buffer**: 
Inside each logic kernel, there may exist buffers to record the its interal state by monitoring the past bit streams, which could be extemely long. 
Those buffers can be counters or shift registers.

5. **Metric Variable**: 
Those are variables to compute specially designed performance metrics.

## Directory Hierarchy
This directory contains four subdirectories, including _'bitstream'_, _'kernel'_,  _'metric'_ and _'test'_, covering three components mentioned above.

### _'bitstream'_ subdirectory
This directory contains the components for **Bit Stream Manipulation**, which deal with the bit stream generation or shuffle for high performance and accuracy.

1. File _gen.py_ is for **Bit Stream Generation**, which refers to generating the bit streams as the input of the computing kernels.
The components currently supported or to be implemented are listed in the table below.

| Name                 | Date          | Reference     | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| BSGen                | Seq. 07, 2019 | [1]           | <ul><li>[x] </li></ul> |
| BSGenMulti           | Nov. 11, 2019 | [1]           | <ul><li>[x] </li></ul> |
| RawScale             | Dec. 07, 2019 | NA            | <ul><li>[x] </li></ul> |
| RNG                  | Seq. 07, 2019 | [2]           | <ul><li>[x] </li></ul> |
| RNGMulti             | Nov. 11, 2019 | [2]           | <ul><li>[x] </li></ul> |
| SourceGen            | Seq. 07, 2019 | [1]           | <ul><li>[x] </li></ul> |


2. File _shuffle.py_ is for **Bit Stream Shuffle**, which refers to shuffling the bit streams for higher computational accuracy. This effect of shuffle can be measured by correlation between bit streams [3].
The components currently supported or to be implemented are listed in the table below.

| Name                 | Date          | Reference     | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| Decorr               |               | [4]           | <ul><li>[ ] </li></ul> |
| DeSync               |               | [4]           | <ul><li>[ ] </li></ul> |
| SkewedSync           | Seq. 07, 2019 | [5]           | <ul><li>[x] </li></ul> |
| Sync                 |               | [4]           | <ul><li>[ ] </li></ul> |


### _'kernel'_ subdirectory
This directory contains the components for **Unary Computing Kernel**, which take bit streams as inputs and perform actual unary computation. The supported kernels are listed as follows.
The components currently supported or to be implemented are listed in the table below.

| Name                 | Date          | Reference     | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| UnaryAdd             | Oct. 10, 2019 | [6]           | <ul><li>[x] </li></ul> |
| UnaryMul             | Nov. 05, 2019 | [6]           | <ul><li>[x] </li></ul> |
| UnaryLinear          | Seq. 27, 2019 | [6]           | <ul><li>[x] </li></ul> |
| UnaryReLU            | Nov. 23, 2019 | [6]           | <ul><li>[x] </li></ul> |
| CORDIV_kernel        | Mar. 08, 2020 | [5]           | <ul><li>[ ] </li></ul> |
| UnaryDiv             | Mar. 08, 2020 | [5]           | <ul><li>[ ] </li></ul> |
| nn_utils             | Nov. 25, 2019 | NA            | <ul><li>[x] </li></ul> |
| GainesAdd            | Mar. 02, 2020 | [1]           | <ul><li>[x] </li></ul> |
| GainesMul            | Dec. 06, 2019 | [1]           | <ul><li>[x] </li></ul> |
| GainesLinear         | Nov. 25, 2019 | [1]           | <ul><li>[x] </li></ul> |
| GainesDiv            | Mar. 08, 2020 | [1]           | <ul><li>[ ] </li></ul> |
| ShiftReg             | Dec. 06, 2019 | NA            | <ul><li>[x] </li></ul> |
| comparison           |               |               | <ul><li>[ ] </li></ul> |
| conv                 |               |               | <ul><li>[ ] </li></ul> |
| exponentiation       |               |               | <ul><li>[ ] </li></ul> |
| max                  |               |               | <ul><li>[ ] </li></ul> |
| min                  |               |               | <ul><li>[ ] </li></ul> |
| pool                 |               |               | <ul><li>[ ] </li></ul> |
| square root          |               | [5]           | <ul><li>[ ] </li></ul> |


### _'metric'_ subdirectory
This directory contains the components for **Performance Metric**, which take bit streams as input and calculate certain performance metrics.
The components currently supported or to be implemented are listed in the table below.

| Name                 | Date          | Reference     | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| Correlation          | Seq. 07, 2019 | [3]           | <ul><li>[x] </li></ul> |
| ProgressivePrecision | Seq. 07, 2019 | [7]           | <ul><li>[x] </li></ul> |
| Stability            | Dec. 27, 2019 | [6]           | <ul><li>[x] </li></ul> |
| NormStability        | Dec. 18, 2019 | NA            | <ul><li>[x] </li></ul> |


### _'test'_ subdirectory
This directory contains simple testing examples for above components, which are integrated with Jupyter-notebook.

## Citation
If you use this work for uGEMM simulation, please cite our paper published in 47th International Symposium on Computer Architecture (ISCA 2020) in which we developed and evaluated uGEMM.

@inproceedings {uGEMM2020isca,
  title = {uGEMM: Unary Computing Architecture for GEMM Applications},
  author = {Di Wu and Jingjie Li and Ruokai Yin and Hsuan Hsiao and Younghyun Kim and Joshua San Miguel},
  booktitle = {Proceedings of the 46th International Symposium on Computer Architecture}
  year = {2020},
}

## Contribution
Active contributor:
1. [Di Wu](https://diwu1990.github.io/)
2. [Ruokai Yin](https://ruokaiyin.github.io/)

Please contact me (di.wu@ece.wisc.edu) if you are interested in contributing to this project!


## Reference
[1] B.R. Gaines, "Stochastic computing systems," in _Advances in Information Systems Science_ 1969.  
[2] S. Liu and J. Han, "Energy efficient stochastic computing with Sobol sequences," in _DATE_ 2017.  
[3] A. Alaghi and J. P. Hayes, "Exploiting correlation in stochastic circuit design," in _ICCD_ 2013.  
[4] V. T. Lee, A. Alaghi and L. Ceze, "Correlation manipulating circuits for stochastic computing," in _DATE_ 2018.  
[5] D. Wu and J. S. Miguel, "In-Stream Stochastic Division and Square Root via Correlation," in _DAC_ 2019.  
[6] D. Wu, etc., "uGEMM: Unary Computing Architecture for GEMM Applications," in _ISCA_ 2020.  
[7] A. Alaghi and J. P. Hayes, "Fast and accurate computation using stochastic circuits," in _DATE_ 2014.  

