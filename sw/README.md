File hierarchy:

1. bitstream:

    peripheral components to support the computing kernels, including:
    
        1.1 bit stream generation/regeneration
        
            from UnaryComputingSim.sw.bitstream.gen import BSGen, BSRegen, SourceGen, RNG
            
            all are tested (Aug. 25, 2019)
            
        1.2 correlation manipulation
        
            from UnaryComputingSim.sw.bitstream.shuffle import SkewedSync, Decorr, Sync, DeSync
            
            SkewedSync is tested (Aug. 25, 2019)
            
2. kernel:

    unary computing units, including:
    
        2.1 addition
        
        2.2 comparison
        
        2.3 conv
        
        2.4 division
        
        2.5 exponentiation
        
        2.6 linear
        
        2.7 maximum
        
        2.8 minimum
        
        2.9 multiplication
        
        2.10 pool
        
        2.11 relu
        
        2.12 square root
        
3. metric:

    measuring metrics:
    
        3.1 correlation
        
            from UnaryComputingSim.sw.metric.metric import Correlation
            
            tested (Aug. 25, 2019)
            
        3.2 progressive precision
        
            from UnaryComputingSim.sw.metric.metric import ProgressivePrecision
            
            tested (Aug. 25, 2019)
            
        3.3 stability
        
            from UnaryComputingSim.sw.metric.metric import Stability
            
            tested (Aug. 25, 2019)
            
        3.4 normalized stability
        
            from UnaryComputingSim.sw.metric.metric import NormStability
            
            tested (Aug. 25, 2019)
        
4. test:

    tests for operators, including simple usage examples
