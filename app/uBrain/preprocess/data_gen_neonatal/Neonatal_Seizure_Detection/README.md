# Neonatal_Seizure_Detection

This repository contains Matlab code for automated neonatal EEG seizure detection algorithms (SDA). 
See instructions_of_use.pdf for more details.

## Algorithms

Neonatal seizure detection algorithms:

Proposed SDA (SDA) - Reference [1] 

Modified Deburchgraeve SDA (SDA_DB_mod) - Reference [1,2,3] (our implementation, trained on our data)

Temko SDA (SDA_T) - Reference [1,4] (our implementation, trained on our data)

Deburchgraeve SDA (SDA_DB) - Reference [1,3] (our implementation)


## EEG file format

EDF format (European Data Format)

see https://www.edfplus.info/specs/edf.html

or

MATLAB formatted data

## Prerequisites

Matlab 2017a

## Description 

This repository contains the Matlab code associated with our recent publication [1]. It contains several algorithms for the detection of neonatal seizure from the EEG. It also contains methods of comparing the algorithm output to the annotation of EEG by the human experts. For details on the see Instructions_For_Use.pdf (to be uploaded).

## Techincal notes

The algorithm includes a notch filter on 50Hz. If the user’s EEG signal has a DC component different to 50Hz, the EEG should be preprocessed with a corresponding notch filter before running the algorithm.

The processing times with the most accurate algorithm (SDA) are long and therefore, users are strongly adviced to use parallelization (optional input n for number of parallel pools)

## Additional Dependencies

The algorithm utilizes an additional external function to show progress of the algorithm:
https://se.mathworks.com/matlabcentral/fileexchange/22161-waitbar-with-time-estimation

## Other

More details on the algorithms and prerequisites can be found in Instructions_For_Use.pdf 

The primary file is neonatal_seizure_detection.m

To reproduce results presented in [1], run original_SDA.m

Files are currently not optimised for efficient processing. Future releases will significantly speed up the runtime.

SVM model files for the three algorithms are available at DOI: 10.5281/zenodo.1281146 or https://zenodo.org/record/1281146#.WxjW7nVubCI.

The database of neonatal EEG used to develop the algorithms is available at DOI: 10.5281/zenodo.2547147 or https://zenodo.org/record/2547147

Single channel annotations have been added to the reproducibility folder.

## Built With

Matlab 2017a

## Authors

Karoliina Tapani and Nathan Stevenson

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References

[1] K. Tapani, S. Vanhatalo and N. Stevenson, Time-varying EEG correlations improve automated neonatal seizure detection, International Journal of Neural Systems. 1850030, 2019

[2] K. Tapani, S. Vanhatalo and N. Stevenson, Incorporating spike correlations into an SVM-based neonatal seizure detector, EMBEC, 2017, pp. 322–325.

[3] W. Deburchgraeve, P. Cherian, M. De Vos, R. Swarte, J. Blok, G. Visser, P. Govaert and S. Van Huffel, Automated neonatal seizure detection mimicking a human observer reading EEG, Clin Neurophysiol 119(11) (2008) 2447–2454.

[4] A. Temko, E. Thomas, W. Marnane, G. Lightbody and G. Boylan, EEG-based neonatal seizure detection with support vector machines, Clin Neurophysiol, 122(3) (2011) 464–473.

## Contact

Karoliina Tapani

Aalto University, Finland

email: karoliina.tapani@aalto.fi

or

Nathan Stevenson

University of Helsinki, Finland and

QIMR Berghofer Medical Research Institute, Australia

email: nathan.stevenson@helsinki.fi

email: nathan.stevenson@QIMRBerghofer.edu.au
