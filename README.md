# hCRF-light

hCRF-light 1.1 (full version http://hcrf.sf.net)
Copyright (C) 2012 Yale Song (yalesong@mit.edu)

This library is a light version of hCRF library (http://hcrf.sf.net)
that contains implementations of HCRF[1] and LDCRF[2]. I added two 
new families of models, multiview counterparts of HCRF and LDCRF [3]
and hierarchical sequence summarization approach of HCRF [4].

The source code is written in C++ using Visual Studio 2008. I tested 
the code on 32 bit and 64 bit versions of Windows 7 machines. I cannot
guarantee its compatibility with other versions of Visual Studio. For 
your convenience, the library comes with precompiled excutables that
work on a 32 bit and 64 bit Windows machines. 

./hCRF-light/readme.md         This file  
./hCRF-light/apps              Contains VS2008 project (hCRF.sln)  
./hCRF-light/apps/matHCRF      Matlab wrapper  
./hCRF-light/apps/testMVLDM    Command-line program  
./hCRF-light/bin               Contains command-line and matlab executables  
./hCRF-light/bin/openMP        Contains command-line and matlab executables (uses openMP)  
./hCRF-light/libs/3rdParty     Contains 3rd party libraries  
./hCRF-light/libs/shared       Contains hCRF library  
./hCRF-light/matlab/           Matlab sample script  
./hCRF-light/toydata/          Toy data in CSV & MAT format  
                                

Usage from command line (Windows 64)  

1. Train and test HCRF with nbHiddenStates = 4  
.\bin\openMP\testModel64MP.exe -m hcrf -h 4 -g 0 -Fd .\toydata\dataTrain.csv -Fq .\toydata\seqLabelsTrain.csv -FD   .\toydata\dataTest.csv -FQ .\toyData\seqLabelsTest.csv  
  
2. Train and test HCNF with nbHiddenStates = 4, nbGates = 4  
.\bin\openMP\testModel64MP.exe -m hcrf -h 4 -g 4 -Fd .\toydata\dataTrain.csv -Fq .\toydata\seqLabelsTrain.csv -FD   .\toydata\dataTest.csv -FQ .\toyData\seqLabelsTest.csv  
  
3. Train and test HSS-HCRF with nbHiddenStates = 4, nbGates = 0, nbFeatureLayers = 4, segmentTau = 0.1  
.\bin\openMP\testModel64MP.exe -m hsshcrf -h 4 -g 0 -HL 4 -HT 0.1 -Fd .\toydata\dataTrain.csv -Fq .\toydata\seqLabelsTrain.csv -FD .\toydata\dataTest.csv -FQ .\toyData\seqLabelsTest.csv  
  
4. Train and test HSS-HCNF with nbHiddenStates = 4, nbGates = 4, nbFeatureLayers = 4, segmentTau = 0.1  
.\bin\openMP\testModel64MP.exe -m hsshcrf -h 4 -g 4 -HL 4 -HT 0.1 -Fd .\toydata\dataTrain.csv -Fq .\toydata\seqLabelsTrain.csv -FD .\toydata\dataTest.csv -FQ .\toyData\seqLabelsTest.csv  
  
See ./hCRF-light/matlab/test.m for examples in Matlab  
  
  
Change log  
ver 2.0 Added Hierarchical Sequence Summarization (HSS) HCRF [4]  
        Changed the name of the project [testMVLDM] to [testModels]  
ver 1.1 Added a project matHCRF, a matlab wrapper  
ver 1.0 Initial release  
  
[1] Ariadna Quattoni, Sy Bor Wang, Louis-Philippe Morency, Michael Collins, Trevor Darrell: Hidden Conditional Random Fields. TPAMI 2007  
  
[2] Louis-Philippe Morency, Ariadna Quattoni, Trevor Darrell: Latent-Dynamic Discriminative Models for Continuous Gesture Recognition. CVPR 2007  

[3] Yale Song, Louis-Philippe Morency, Randall Davis: Multi-View Latent Variable Discriminative Models for Action Recognition. CVPR 2012  

[4] Yale Song, Louis-Philippe Morency, Randall Davis: Action Recognition by Hierarchical Sequence Summarization. CVPR 2013  
