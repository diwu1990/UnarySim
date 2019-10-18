# Overview
This directory contains the required components for cycle-accurate simulation for Unary Computing based on PyTorch, where the simulation can be done on either CPU or GPU.
Those components includes _**Bit Stream Manipulation**_, _**Unary Computing Kernel**_ and _**Performance Metric**_.

## Data Representation
We have three categories of data in UnarySim:
1. **Bit Stream**: At each cycle, the bit in each bit stream hysically flows through cascaded logic kernels. Those bits count for most of the memory space. For the sake of efficiency, they are forced to be _**torch.int8**_ type.
2. **Bit Buffer**: Inside each logic kernel, there may exist some buffers to record the its interal state. Those buffers can be counters or shift registers. Those buffers need to record past bit streams, which could be extemely long. To ensure the functionality, they are forced to be _**torch.long**_ type.
3. **Metric Variable**: Those are variables to compute specially designed metrics, which are usually floating point values. To provide precise records of target metrics, they are forced to be _**torch.float**_ type.

## Directory Hierarchy
This directory contains four subdirectories, including _**bitstream**_, _**kernel**_,  _**metric**_ and _**test**_, corresponding to the three components mentioned above.

### _bitstream_ subdirectory
This directory contains the components for _**Bit Stream Manipulation**_, which deal with the bit stream generation or shuffle for high performance and accuracy.

_gen.py_ is for _**Bit Stream Generation**_.

(from UnaryComputingSim.sw.bitstream.gen import \*)

Status | Name                 | Date          | Note
------ | -------------------- | ------------- | -------------
- [x]  | BSGen                | Seq. 7, 2019  | 
- [ ]  | BSRegen              |               | 
- [x]  | SourceGen            | Seq. 7, 2019  | 
- [x]  | RNG                  | Seq. 7, 2019  | 

_shuffle.py_ is for _**Bit Stream Shuffle**_.

(from UnaryComputingSim.sw.bitstream.shuffle import \*)

Status | Name                 | Date          | Note
------ | -------------------- | ------------- | -------------
- [ ]  | Decorr               |               | 
- [ ]  | DeSync               |               | 
- [x]  | SkewedSync           | Seq. 7, 2019  | 
- [ ]  | Sync                 |               | 

### _kernel_ subdirectory
This directory contains the components for _**Unary Computing Kernel**_, which take bit streams as inputs and perform actual unary computation. The supported kernels are listed as follows.

Status | Name                 | Date          | Note
------ | -------------------- | ------------- | -------------
- [ ]  | addition             |               | 
- [ ]  | comparison           |               | 
- [ ]  | conv                 |               | 
- [ ]  | division             |               | 
- [ ]  | exponentiation       |               | 
- [x]  | linear               | Seq. 27, 2019 | 
- [ ]  | max                  |               | 
- [ ]  | min                  |               | 
- [ ]  | multiplication       |               | 
- [ ]  | pool                 |               | 
- [ ]  | relu                 |               | 
- [ ]  | square root          |               | 

### _metric_ subdirectory
This directory contains the components for  _**Performance Metric**_, which take bit streams as inputs and calculate certain performance metrics.

(from UnaryComputingSim.sw.metric.metric import \*)

Status | Name                 | Date          | Note
------ | -------------------- | ------------- | -------------
- [x]  | Correlation          | Seq. 7, 2019  | 
- [x]  | ProgressivePrecision | Seq. 7, 2019  | 
- [x]  | Stability            | Seq. 7, 2019  | 
- [x]  | NormStability        | Seq. 7, 2019  | 
        
### _test_ subdirectory
This directory contains the examples for testing for above implemented kernels.
