## Unary Computing
_**Unary computing**_ is a type of computing paradigm based on serial data in streams, rather than parallel data in conventional binary computing. 
The unary computing kernels are much simpler than conventional binary counterparts, but spend more cycles for a single operation. 

In general, a single unary computing kernel may not outperform the binary in terms of energy efficiency. However, at the system level, there are two advantages in unary computing over binary computing with respect to the system input.
1. If the system input is analog data from sensors, unary computing eliminates the process of converting the analog signals to binary format and storing it, reducing the energy consumption [1, 2].
2. If the system input is already binary, due to the higher computing density resulting from simple logic, unary computing can achieve even higher energy efficiency than binary computing when the computational intensity reaches a certain threshold [3].

## UnarySim
This _**UnarySim**_ is a PyTorch-based cycle-accurate simulator for large scale unary computing, compatible to both CPU and GPU with high efficiency.

### _'sw'_ subdirectory
This folder includes the software implementation of multiple unary operators. Please check inside for how to use _**UnarySim**_.

### _'hw'_ subdirectory
This folder includes the hardware implementation of operators in _'sw'_ subdirectory, and is still in progress.

## Reference
[1] V. T. Lee, A. Alaghi, J. P. Hayes, V. Sathe, and L. Ceze, "Energy-efficient hybrid stochastic-binary neural networks for near-sensor computing", in *DATE* 2017.

[2] S. K. Khatamifard, M. H. Najafi, A. Ghoreyshi, U. R. Karpuzcu and D. J. Lilja, "On Memory System Design for Stochastic Computing", in *IEEE Computer Architecture Letters* 2018.

[3] V. T. Lee, A. Alaghi, R. Pamula, V. S. Sathe, L. Ceze and M. Oskin, "Architecture Considerations for Stochastic Computing Accelerators", in *TCAD* 2018.  

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

