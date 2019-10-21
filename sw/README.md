# Overview

This directory contains the required components for cycle-accurate software simulation for unary computing. UnarySim is based on [PyTorch](https://pytorch.org/), a deep learning framework from Facebook, such that the simulation can be done on either CPU or GPU efficiently.

The components included in UnarySim belong to three categories, including 
1. **Bit Stream Manipulation**
2. **Unary Computing Kernel**
3. **Performance Metric**

Among those, components in **Bit Stream Manipulation** and **Unary Computing Kernel** can pysically exist in real hardware, while those from **Performance Metric** is usually for performance analysis.

## Prerequisite
1. PyTorch version >= 1.0
2. Python version >= 3.0

## Data Representation
UnarySim has five categories of data, with each having preferred data type in PyTorch.

1. **Source Data**: 
The input source data in unary computing need to be ranging from _0_ to _1_ in **unipolar** format, or from _-1_ to _1_ in **bipolar** format. 
The input source data (_source_) is scaled to a certain range (as in _unipolar/bipolar_ format) from the raw data (_raw_) with respect to its maximum.
More specifically, such a relationship is formulated as _source = raw / max(raw)_. Thus, the type of _source data_ is suggested to be _**'torch.float'**_.

2. **Random Number**: 
The random numbers (_rand_) are to be compared with the source data in order to generate the bit streams. To notice, source data is in _unipolar/bipolar_ format, while random numbers are integers. 
To compare them, source data requires to scale up by the _bitwidth_ of random numbers. 
At each cycle, if the _round(source * 2^bitwidth) > rand_, a bit of logic 1 in the bit stream will be generated; otherwise, a bit of logic 0 will be generated. 
To support sufficiently long bit streams, the type of _random number_ is suggested to be _**'torch.long'**_.

3. **Bit Stream**: 
At each cycle, the bits in bit streams physically flow through cascaded logic kernels, and they count for most of the memory space during simulation. 
For the sake of execution efficiency, the type of _bit stream_ is suggested to be _**'torch.int8'**_.

4. **Bit Buffer**: 
Inside each logic kernel, there may exist buffers to record the its interal state by monitoring the past bit streams, which could be extemely long. 
Those buffers can be counters or shift registers. 
To ensure the correctness (not overflowing), the type of _bit buffer_ is suggested to be _**'torch.long'**_.

5. **Metric Variable**: 
Those are variables to compute specially designed performance metrics, which are usually floating point values. 
To provide precise records of target metrics, the type of _metric variable_ is suggested to be _**'torch.float'**_.

## Directory Hierarchy
This directory contains four subdirectories, including _'bitstream'_, _'kernel'_,  _'metric'_ and _'test'_, corresponding to the three components mentioned above.

### _'bitstream'_ subdirectory
This directory contains the components for **Bit Stream Manipulation**, which deal with the bit stream generation or shuffle for high performance and accuracy.

1. File _gen.py_ is for **Bit Stream Generation**, which refers to generating the bit streams as the input of the computing kernels.
The components currently supported or to be implemented are listed in the table below.

| Name                 | Date          | Reference     | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| BSGen                | Seq. 7, 2019  | [1]           | <ul><li>[x] </li></ul> |
| BSReGen              |               | NA            | <ul><li>[ ] </li></ul> |
| RawScale             | Oct. 10, 2019 | NA            | <ul><li>[x] </li></ul> |
| RNG                  | Seq. 7, 2019  | [2]           | <ul><li>[x] </li></ul> |
| SourceGen            | Seq. 7, 2019  | [1]           | <ul><li>[x] </li></ul> |


2. File _shuffle.py_ is for **Bit Stream Shuffle**, which refers to shuffling the bit streams for higher computational accuracy. This effect of shuffle can be measured by correlation between bit streams [3].
The components currently supported or to be implemented are listed in the table below.

| Name                 | Date          | Reference     | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| Decorr               |               | [4]           | <ul><li>[ ] </li></ul> |
| DeSync               |               | [4]           | <ul><li>[ ] </li></ul> |
| SkewedSync           | Seq. 7, 2019  | [5]           | <ul><li>[x] </li></ul> |
| Sync                 |               | [4]           | <ul><li>[ ] </li></ul> |


### _'kernel'_ subdirectory
This directory contains the components for **Unary Computing Kernel**, which take bit streams as inputs and perform actual unary computation. The supported kernels are listed as follows.
The components currently supported or to be implemented are listed in the table below.

| Name                 | Date          | Reference     | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| UnaryAdd             | Oct. 10, 2019 | [6]           | <ul><li>[x] </li></ul> |
| comparison           |               |               | <ul><li>[ ] </li></ul> |
| conv                 |               |               | <ul><li>[ ] </li></ul> |
| division             |               | [5]           | <ul><li>[ ] </li></ul> |
| exponentiation       |               |               | <ul><li>[ ] </li></ul> |
| linear               | Seq. 27, 2019 | [6]           | <ul><li>[x] </li></ul> |
| max                  |               |               | <ul><li>[ ] </li></ul> |
| min                  |               |               | <ul><li>[ ] </li></ul> |
| multiplication       |               | [6]           | <ul><li>[ ] </li></ul> |
| pool                 |               |               | <ul><li>[ ] </li></ul> |
| relu                 |               |               | <ul><li>[ ] </li></ul> |
| square root          |               | [5]           | <ul><li>[ ] </li></ul> |


### _'metric'_ subdirectory
This directory contains the components for **Performance Metric**, which take bit streams as inputs and calculate certain performance metrics.
The components currently supported or to be implemented are listed in the table below.

| Name                 | Date          | Reference     | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| Correlation          | Seq. 7, 2019  | [3]           | <ul><li>[x] </li></ul> |
| ProgressivePrecision | Seq. 7, 2019  | [7]           | <ul><li>[x] </li></ul> |
| Stability            | Seq. 7, 2019  | [6]           | <ul><li>[x] </li></ul> |
| NormStability        | Seq. 7, 2019  | NA            | <ul><li>[x] </li></ul> |


### _'test'_ subdirectory
This directory contains simple testing examples for above components, which are integrated with Jupyter-notebook.


## Contribution
Please contact me if you are interested in contributing to this project!

## Reference
[1] B.R. Gaines, "Stochastic computing systems," in _Advances in Information Systems Science_ 1969.  
[2] S. Liu and J. Han, "Energy efficient stochastic computing with Sobol sequences," in _DATE_ 2017.  
[3] A. Alaghi and J. P. Hayes, "Exploiting correlation in stochastic circuit design," in _ICCD_ 2013.  
[4] V. T. Lee, A. Alaghi and L. Ceze, "Correlation manipulating circuits for stochastic computing," in _DATE_ 2018.  
[5] D. Wu and J. S. Miguel, "In-Stream Stochastic Division and Square Root via Correlation," in _DAC_ 2019.  
[6] D. Wu, etc., "uGEMM: Unary Computing Architecture for GEMM Applications," submitted for review.  
[7] A. Alaghi and J. P. Hayes, "Fast and accurate computation using stochastic circuits," in _DATE_ 2014.  

