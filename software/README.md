File hierarchy:

1. components:

    peripheral components to support the computing kernels, including:
    
        1.1 bit stream generation/regeneration
        
            from UnaryComputingSim.software.components.bsgen import BSGen, BSRegen, SourceGen
            
            all are tested (Aug. 25, 2019)
            
        1.2 random number generation
        
            from UnaryComputingSim.software.components.rng import RNG
            
            tested (Aug. 25, 2019)
            
        1.3 correlation manipulation
        
            from UnaryComputingSim.software.components.bitshuffle import SkewedSync, Decorr, Sync, DeSync
            
            SkewedSync is tested (Aug. 25, 2019)
            
2. kernels:

    unary computing units, including:
    
        2.1 addition
        
        2.2 multiplication
        
        2.3 division
        
        2.4 exponentiation
        
        2.5 comparison
        
        2.6 maximum
        
        2.7 minimum
        
        2.8 square root
        
        2.9 relu
        
        2.10 linear layer
        
        2.11 conv2d layer
        
3. metric:

    measuring metrics:
    
        3.1 correlation
        
            from UnaryComputingSim.software.metric.metric import Correlation
            
            tested (Aug. 25, 2019)
            
        3.2 progressive precision
        
            from UnaryComputingSim.software.metric.metric import ProgressivePrecision
            
            tested (Aug. 25, 2019)
            
        3.3 stability
        
            from UnaryComputingSim.software.metric.metric import Stability
            
            tested (Aug. 25, 2019)
            
        3.4 normalized stability
        
            from UnaryComputingSim.software.metric.metric import NormStability
            
            tested (Aug. 25, 2019)
        
4. test:

    tests for operators, including simple usage examples
