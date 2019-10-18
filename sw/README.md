# Overview
This directory contains the required components for cycle-accurate simulation for Unary Computing based on PyTorch, where the simulation can be done on either CPU or GPU.
Those components include _**Bit Stream Manipulation**_, _**Unary Computing Kernel**_ and _**Performance Metric**_.

## Data Representation
UnarySim has three categories of data, with each having dedicated data type in PyTorch.
1. **Bit Stream**: At each cycle, the bits in bit streams physically flow through cascaded logic kernels, and they count for most of the memory space. For the sake of execution efficiency, they are forced to be _**'torch.int8'**_ type.
2. **Bit Buffer**: Inside each logic kernel, there may exist some buffers to record the its interal state by monitoring the past bit streams, which could be extemely long. Those buffers can be counters or shift registers. To ensure the correctness (not overflowing), they are forced to be _**'torch.long'**_ type.
3. **Metric Variable**: Those are variables to compute specially designed performance metrics, which are usually floating point values. To provide precise records of target metrics, they are forced to be _**'torch.float'**_ type.

## Directory Hierarchy
This directory contains four subdirectories, including _**'bitstream'**_, _**'kernel'**_,  _**'metric'**_ and _**'test'**_, corresponding to the three components mentioned above.

### _'bitstream'_ subdirectory
This directory contains the components for _**Bit Stream Manipulation**_, which deal with the bit stream generation or shuffle for high performance and accuracy.

1. _gen.py_ is for _**Bit Stream Generation**_.

(from UnaryComputingSim.sw.bitstream.gen import \*)

| Name                 | Date          | Note          | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| BSGen                | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| BSRegen              |               |               | <ul><li>[ ] </li></ul> |
| SourceGen            | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| RNG                  | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |


2. _shuffle.py_ is for _**Bit Stream Shuffle**_.

(from UnaryComputingSim.sw.bitstream.shuffle import \*)

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

(from UnaryComputingSim.sw.metric.metric import \*)

| Name                 | Date          | Note          | Status                 |
| -------------------- | ------------- | ------------- | ---------------------- |
| Correlation          | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| ProgressivePrecision | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| Stability            | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |
| NormStability        | Seq. 7, 2019  |               | <ul><li>[x] </li></ul> |


### _'test'_ subdirectory
This directory contains the testing examples for above implemented kernels.
