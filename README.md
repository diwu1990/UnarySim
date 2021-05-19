## Unary Computing
[_**Unary computing**_](https://unarycomputing.github.io/) is a type of computing paradigm based on serial bits in streams, rather than parallel bits in conventional binary computing. 
The unary computing kernels are much simpler than conventional binary counterparts, but spend more cycles for a single operation. 

In general, a single unary computing kernel may not outperform the binary counterpart in terms of energy efficiency. However, at the system level, there are two advantages in unary computing over binary computing with respect to the system input.
1. If the system input is analog from sensors, unary computing eliminates the process of converting the analog signals to binary format and storing it, reducing the energy consumption [1, 2].
2. If the system input is already binary, due to the higher computing density resulting from simple logic, unary computing can achieve even higher energy efficiency than binary computing when the computational intensity reaches a certain threshold [3].

## UnarySim
This _**UnarySim**_ is a PyTorch-based cycle-accurate simulator for large scale unary computing, compatible to both CPU and GPU with high efficiency.
The components included in UnarySim belong to three categories, including 
1. **Stream Manipulation**
2. **Computing Kernel**
3. **Performance Metric**

Among those, components in **Stream Manipulation** and **Computing Kernel** can physically exist in real hardware, while those in **Performance Metric** are virtually for performance analysis.

### Prerequisite
1. PyTorch (Version >= 1.7)
2. Python (Version >= 3.6)
3. [pylfsr](https://github.com/Nikeshbajaj/Linear_Feedback_Shift_Register)

### Data Representation
UnarySim has five categories of data, with each having default data type as _**'torch.float'**_ in PyTorch.

1. **Source Data**: 
The input source data in unary computing need to be ranging from _0_ to _1_ in **unipolar** format (Uni), or from _-1_ to _1_ in **bipolar** format (Bi). 
The input source data (_source_) is scaled to a certain range (as in _unipolar/bipolar_ format) from the raw data (_raw_) with respect to its maximum.
An example scaling is _source = raw / max(raw)_.

2. **Random Number**: 
The random numbers (_rand_) are to be compared with the source data in order to generate the bitstreams. To notice, source data is in _unipolar/bipolar_ format, while random numbers are integers. 
To compare them, source data requires to scale up by the _bitwidth_ of random numbers. 
At each cycle, if _round(source * 2^bitwidth) > rand_, a bit of logic 1 in the stream will be generated; otherwise, a bit of logic 0 will be generated.

3. **Stream**: 
At each cycle, the bitstreams physically flow through cascaded logic kernels, and they count for most of the memory space during simulation. 
The bitstream can leverage either rate coding (RC) or temporal coding (TC).

4. **Buffer**: 
Inside each logic kernel, there may exist buffers, like counters or shift registers, to compute future data by monitoring the past bitstreams. 

5. **Metric Variable**: 
Those are variables to compute specially designed performance metrics.

### Directory Hierarchy
This directory contains multiple subdirectories, including _'stream'_, _'kernel'_, _'metric'_, _'test'_, _'app'_, _'hw'_,  covering three components mentioned above.

#### _'stream'_ subdirectory
This directory contains the components for **Stream Manipulation**, which manipulate the bitstreams for high performance and accuracy.

| Name                 | Date         | Encoding | Polarity | Reference | Status                 |
| -------------------- | ------------ | -------- | -------- | --------- | ---------------------- |
| BSGen                | Seq 07, 2019 | Both     | Both     | [4]       | <ul><li>[x] </li></ul> |
| BSGenMulti           | Nov 11, 2019 | Both     | Both     | [4]       | <ul><li>[x] </li></ul> |
| RawScale             | Dec 07, 2019 | Both     | Either   | NA        | <ul><li>[x] </li></ul> |
| RNG                  | Seq 07, 2019 | Both     | Both     | [5]       | <ul><li>[x] </li></ul> |
| RNGMulti             | Nov 11, 2019 | Both     | Both     | [5]       | <ul><li>[x] </li></ul> |
| SourceGen            | Seq 07, 2019 | Both     | Either   | [4]       | <ul><li>[x] </li></ul> |
| Decorr               |              | Both     | Both     | [6]       | <ul><li>[ ] </li></ul> |
| DeSync               |              | RC       | Both     | [6]       | <ul><li>[ ] </li></ul> |
| SkewedSync           | Seq 07, 2019 | Both     | Both     | [7]       | <ul><li>[x] </li></ul> |
| Sync                 |              | Both     | Both     | [6]       | <ul><li>[ ] </li></ul> |
| Bi2Uni               | Mar 31, 2020 | RC       | Bi       | [13]      | <ul><li>[x] </li></ul> |
| Uni2Bi               | Mar 31, 2020 | RC       | Uni      | [13]      | <ul><li>[x] </li></ul> |

#### _'kernel'_ subdirectory
This directory contains the components for **Computing Kernel**, which take bitstreams as inputs and perform actual unary computation. The supported kernels are listed as follows.
The components currently supported or to be implemented are listed in the table below.

| Name                 | Date         | Encoding | Polarity | Reference | Status                 |
| -------------------- | ------------ | -------- | -------- | --------- | ---------------------- |
| UnaryAbs             | Mar 25, 2020 | RC       | Bi       | NA        | <ul><li>[x] </li></ul> |
| UnarySign            | Mar 31, 2020 | RC       | Bi       | NA        | <ul><li>[x] </li></ul> |
| UnaryAdd             | Oct 10, 2019 | Both     | Either   | [8]       | <ul><li>[x] </li></ul> |
| UnaryMul             | Nov 05, 2019 | Both     | Either   | [8]       | <ul><li>[x] </li></ul> |
| UnaryLinear          | Seq 27, 2019 | Both     | Either   | [8]       | <ul><li>[x] </li></ul> |
| UnaryReLU            | Nov 23, 2019 | Either   | Either   | [8]       | <ul><li>[x] </li></ul> |
| CORDIV_kernel        | Mar 08, 2020 | RC       | Uni      | [5]       | <ul><li>[x] </li></ul> |
| UnaryDiv             | Apr 01, 2020 | RC       | Either   | [5]       | <ul><li>[x] </li></ul> |
| UnarySqrt            | Apr 02, 2020 | RC       | Either   | [5]       | <ul><li>[x] </li></ul> |
| GainesAdd            | Mar 02, 2020 | Both     | Either   | [4]       | <ul><li>[x] </li></ul> |
| GainesMul            | Dec 06, 2019 | RC       | Either   | [4]       | <ul><li>[x] </li></ul> |
| GainesLinear         | Nov 25, 2019 | RC       | Either   | [4]       | <ul><li>[x] </li></ul> |
| GainesDiv            | Mar 08, 2020 | RC       | Either   | [4]       | <ul><li>[x] </li></ul> |
| GainesSqrt           | Mar 24, 2020 | RC       | Either   | [4]       | <ul><li>[x] </li></ul> |
| nn_utils             | Nov 25, 2019 | NA       | NA       | NA        | <ul><li>[x] </li></ul> |
| JKFF                 | Apr 01, 2020 | NA       | NA       | NA        | <ul><li>[x] </li></ul> |
| ShiftReg             | Dec 06, 2019 | NA       | NA       | NA        | <ul><li>[x] </li></ul> |
| comparison           |              |          |          |           | <ul><li>[ ] </li></ul> |
| conv                 |              |          |          |           | <ul><li>[ ] </li></ul> |
| exponentiation       |              |          |          |           | <ul><li>[ ] </li></ul> |
| max                  |              |          |          |           | <ul><li>[ ] </li></ul> |
| min                  |              |          |          |           | <ul><li>[ ] </li></ul> |
| pool                 |              |          |          |           | <ul><li>[ ] </li></ul> |

#### _'metric'_ subdirectory
This directory contains the components for **Performance Metric**, which take bit streams as input and calculate certain performance metrics.
The components currently supported or to be implemented are listed in the table below.

| Name                 | Date         | Encoding | Polarity | Reference | Status                 |
| -------------------- | ------------ | -------- | -------- | --------- | ---------------------- |
| Correlation          | Seq 07, 2019 | Both     | Both     | [9]       | <ul><li>[x] </li></ul> |
| ProgressiveError     | Seq 07, 2019 | Both     | Either   | [10]      | <ul><li>[x] </li></ul> |
| Stability            | Dec 27, 2019 | Both     | Either   | [8]       | <ul><li>[x] </li></ul> |
| NormStability        | Dec 18, 2019 | Both     | Either   | [12]      | <ul><li>[x] </li></ul> |
| NSBuilder            | Mar 31, 2020 | Both     | Either   | [12]      | <ul><li>[x] </li></ul> |

#### _'test'_ subdirectory
This directory contains simple testing examples for above components.

#### _'app'_ subdirectory
This directory contains several applications implemented using this UnarySim.

#### _'hw'_ subdirectory
This directory includes the hardware implementation of components in **Stream Manipulation** and **Computing Kernel**, and is still in progress.

## Reference
[1] V. T. Lee, A. Alaghi, J. P. Hayes, V. Sathe, and L. Ceze, "Energy-efficient hybrid stochastic-binary neural networks for near-sensor computing", in _DATE_ 2017.
[2] S. K. Khatamifard, M. H. Najafi, A. Ghoreyshi, U. R. Karpuzcu and D. J. Lilja, "On Memory System Design for Stochastic Computing", in _IEEE Computer Architecture Letters_ 2018.
[3] V. T. Lee, A. Alaghi, R. Pamula, V. S. Sathe, L. Ceze and M. Oskin, "Architecture Considerations for Stochastic Computing Accelerators", in _TCAD_ 2018.  
[4] B.R. Gaines, "Stochastic computing systems," in _Advances in Information Systems Science_ 1969.  
[5] S. Liu and J. Han, "Energy efficient stochastic computing with Sobol sequences," in _DATE_ 2017.  
[6] V. T. Lee, A. Alaghi and L. Ceze, "Correlation manipulating circuits for stochastic computing," in _DATE_ 2018.  
[7] D. Wu and J. S. Miguel, "In-Stream Stochastic Division and Square Root via Correlation," in _DAC_ 2019.  
[8] D. Wu, J. Li, R. Yin, H. Hsiao, Y. Kim and J. San Miguel, "uGEMM: Unary Computing Architecture for GEMM Applications," in _ISCA_ 2020.  
[9] A. Alaghi and J. P. Hayes, "Exploiting correlation in stochastic circuit design," in _ICCD_ 2013.  
[10] A. Alaghi and J. P. Hayes, "Fast and accurate computation using stochastic circuits," in _DATE_ 2014.  
[11] P. Li and D. J. Lilja, W. Qian, M. D. Riedel and K. Bazargan, "Logical Computation on Stochastic Bit Streams with Linear Finite-State Machines," in _IEEE Transactions On Computers_ 2014.  
[12] D. Wu, R. Yin and J. San Miguel, "Normalized Stability: A Cross-Level Design Metric for Early Termination in Stochastic Computing", in _ASP-DAC_ 2021.  
[13] D. Wu, R. Yin and J. San Miguel, "In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing", in _IEEE Design & Test_ 2021.

## Citing
If you find UnarySim or uGEMM is useful for your research, please use the following bibtex to cite us,

```
@inproceedings{diwu2020uGEMM,
  title = {{uGEMM: Unary Computing Architecture for GEMM Applications}},
  author = {Di Wu and Jingjie Li and Ruokai Yin and Hsuan Hsiao and Younghyun Kim and Joshua San Miguel},
  booktitle = {Proceedings of the 46th International Symposium on Computer Architecture},
  year = {2020},
}
```

## Contribution
Active contributor:
1. [Di Wu](https://diwu1990.github.io/)
2. [Ruokai Yin](https://ruokaiyin.github.io/)

Please contact me (di.wu@ece.wisc.edu) if you are interested in contributing to this project!

