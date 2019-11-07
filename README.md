## Unary Computing
_**Unary computing**_ is a type of computing paradigm based on serial bits in streams, rather than parallel data in conventional binary computing. 
The unary computing kernels are much simpler than conventional binary counterparts, but spend more cycles for a single operation. 

In general, a single unary computing kernel may not outperform the binary in terms of energy efficiency. However, at the system level, there are two advantages in unary computing over binary computing with respect to the system input.
1. If the system input is analog data from sensors, unary computing eliminates the process of converting the analog signals to binary format and storing it, reducing the energy consumption [1].
2. If the system input is already binary, due to the higher computing density resulting from simple logic, unary computing can achieve even higher energy efficiency than binary computing when the computing density reaches a certain threshold [2].

## UnarySim
This _**UnarySim**_ is a PyTorch-based cycle-accurate simulator for large scale unary computing.

## Reference
[1] V. T. Lee, A. Alaghi, J. P. Hayes, V. Sathe, and L. Ceze, "Energy-efficient hybrid stochastic-binary neural networks for near-sensor computing", in *DATE* 2017.

[2] V. T. Lee, A. Alaghi, R. Pamula, V. S. Sathe, L. Ceze and M. Oskin, "Architecture Considerations for Stochastic Computing Accelerators", in *TCAD* 2018.  
