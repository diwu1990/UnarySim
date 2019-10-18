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

There are two files in this directory, _gen.py_ for _**Bit Stream Generation**_ and _shuffle.py_ for _**Bit Stream Shuffle**_.
    
1. bit stream generation/regeneration
    from UnaryComputingSim.sw.bitstream.gen import BSGen, BSRegen, SourceGen, RNG
    BSGen, SourceGen and RNG are implemented and tested (Seq. 7, 2019)

2. correlation manipulation
    from UnaryComputingSim.sw.bitstream.shuffle import SkewedSync, Decorr, Sync, DeSync
    SkewedSync is implemented and tested (Seq. 7, 2019)
            
### _kernel_ subdirectory
This directory contains the components for _**Unary Computing Kernel**_, which take bit streams as inputs and perform actual computation. The kernels are listed as follows, where [x] means the kernel is implemented and tested.

Name | Status | Date
------------ | ------------- | -------------
addition | [] | 
comparison | [] | 
conv | [] | 
division | [] | 
exponentiation | [] | 
linear | [x] | Seq. 27, 2019
max | [] | 
min | [] | 
multiplication | [] | 
pool | [] | 
relu | [] | 
square root | [] | 
        
### _metric_ subdirectory

    measuring metrics:
    
        3.1 correlation
            from UnaryComputingSim.sw.metric.metric import Correlation
            tested (Seq. 7, 2019)
            
        3.2 progressive precision
            from UnaryComputingSim.sw.metric.metric import ProgressivePrecision
            tested (Seq. 7, 2019)
            
        3.3 stability
            from UnaryComputingSim.sw.metric.metric import Stability
            tested (Seq. 7, 2019)
            
        3.4 normalized stability
            from UnaryComputingSim.sw.metric.metric import NormStability
            tested (Seq. 7, 2019)
        
4. test:

    tests for above components
