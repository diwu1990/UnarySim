# Overview
This directory contains the required components for cycle-accurate software simulation for _**Unary Computing**_ based on PyTorch, where the simulation can be done on either CPU or GPU. The unary computing is a new computing paradigm based on bit streams.
The components in this include _**Bit Stream Manipulation**_, _**Unary Computing Kernel**_ and _**Performance Metric**_.

## Data Representation
UnarySim has five categories of data, with each having dedicated data type in PyTorch.
1. **Source Data**: The source input data for unary computing ranges from _0_ to _1_ in _**unipolar**_ format, or from _-1_ to _1_ in _**bipolar**_ format. The input data (_source_) is scaled to a certain range (as in _unipolar/bipolar_ format) from the original value (_origin_) with respect to maximum origin. More specifically, such relationship is formulated as _source = origin / max(origin). Thus, source data type is suggested to be _**'torch.float'**_.
2. **Random Number**: The random numbers (_rand_) are to be compared with the source data in order to generate the bit stream. To notice, source data is in _unipolar/bipolar_ format, while random numbers are integers. To compare them, source data requires to scale up by the _bitwidth_ of random numbers. At each cycle, if the _round(source * 2^bitwidth) > rand_, a bit 1 in the bit stream will be generated; otherwise, a bit 0 will be generated. To support sufficiently long bit streams, the random number type is suggested to be _**'torch.long'**_.
3. **Bit Stream**: At each cycle, the bits in bit streams physically flow through cascaded logic kernels, and they count for most of the memory space for simulation. For the sake of execution efficiency, bit stream type is suggested to be _**'torch.int8'**_.
4. **Bit Buffer**: Inside each logic kernel, there may exist some buffers to record the its interal state by monitoring the past bit streams, which could be extemely long. Those buffers can be counters or shift registers. To ensure the correctness (not overflowing), bit buffer type is suggested to be _**'torch.long'**_.
5. **Metric Variable**: Those are variables to compute specially designed performance metrics, which are usually floating point values. To provide precise records of target metrics, metric variable type is suggested to be _**'torch.float'**_.

## Directory Hierarchy
This directory contains four subdirectories, including _**'bitstream'**_, _**'kernel'**_,  _**'metric'**_ and _**'test'**_, corresponding to the three components mentioned above.

### _'bitstream'_ subdirectory
This directory contains the components for _**Bit Stream Manipulation**_, which deal with the bit stream generation or shuffle for high performance and accuracy.

1. _gen.py_ is for _**Bit Stream Generation**_. The _**Bit Stream Generation**_ refers to how to generate the bit streams to be fed into the computing kernels.

(from UnarySim.sw.bitstream.gen import \*)

| Name                 | Date          | Note          | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| BSGen                | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| BSRegen              |               |               | <ul><li>[ ] </li></ul> |
| SourceGen            | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| SourceScale          |               |               | <ul><li>[ ] </li></ul> |
| RNG                  | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |


2. _shuffle.py_ is for _**Bit Stream Shuffle**_.

(from UnarySim.sw.bitstream.shuffle import \*)

| Name                 | Date          | Note          | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| Decorr               |               |               | <ul><li>[ ] </li></ul> |
| DeSync               |               |               | <ul><li>[ ] </li></ul> |
| SkewedSync           | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| Sync                 |               |               | <ul><li>[ ] </li></ul> |


### _'kernel'_ subdirectory
This directory contains the components for _**Unary Computing Kernel**_, which take bit streams as inputs and perform actual unary computation. The supported kernels are listed as follows.

| Name                 | Date          | Note          | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| addition             |               |               | <ul><li>[ ] </li></ul> |
| comparison           |               |               | <ul><li>[ ] </li></ul> |
| conv                 |               |               | <ul><li>[ ] </li></ul> |
| division             |               |               | <ul><li>[ ] </li></ul> |
| exponentiation       |               |               | <ul><li>[ ] </li></ul> |
| linear               | Seq. 27, 2019 |               | <ul><li>[x] </li></ul> |
| max                  |               |               | <ul><li>[ ] </li></ul> |
| min                  |               |               | <ul><li>[ ] </li></ul> |
| multiplication       |               |               | <ul><li>[ ] </li></ul> |
| pool                 |               |               | <ul><li>[ ] </li></ul> |
| relu                 |               |               | <ul><li>[ ] </li></ul> |
| square root          |               |               | <ul><li>[ ] </li></ul> |


### _'metric'_ subdirectory
This directory contains the components for  _**Performance Metric**_, which take bit streams as inputs and calculate certain performance metrics.

(from UnarySim.sw.metric.metric import \*)

| Name                 | Date          | Note          | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| Correlation          | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| ProgressivePrecision | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| Stability            | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| NormStability        | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |


### _'test'_ subdirectory
This directory contains the testing examples for above implemented kernels.


Please contact me if you are interested in contributing to this project!
