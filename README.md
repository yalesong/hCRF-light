# hCRF-light

hCRF-light Version 3.0

This library is a light version of hCRF library (http://hcrf.sf.net)
that contains implementations of HCRF[1] and LDCRF[2]. I added three
new families of models, multiview counterparts of HCRF and LDCRF [3],
hierarchical sequence summarization approach of HCRF [4], and one-class
formulation of both CRF and HCRF [5].

## INSTALLATION

Follow instructions in ./compile.sh 

## TEST RUN

If you compiled the library without any error, it will produce a binary
at ./distribute/bin/hcrf-light. You can use the binary to try a variety
of functionalities provided by the hCRF-light library.

First, run the following commmand to train and test an HCRF with toy data:
```
./distribute/bin/hcrf-light -m hcrf -h 4 -s 10 \
-Fd ./data/toy/dataTrain.csv -Fq ./data/toy/seqLabelsTrain.csv \
-FD ./data/toy/dataTest.csv -FQ ./data/toy/seqLabelsTest.csv
```
If you're facing "Cannot find shared library" error, please refer this to [fix](https://github.com/yalesong/hCRF-light/issues/3) that.

It trains an HCRF (-m hcrf) with 4 hidden states (-h 4) and the L2
regularization factor set at 10 (-s 10). It will train the model using
data and label files passed by the parameters -Fd and -Fq, respectively;
once the training is finished, it will then test the model using data 
and label files passed by the parameters -FD and -FQ. 

Once the process is terminated, it will create four result files: 
- results.txt
- stats.txt
- model.txt
- features.txt

The first two (results.txt and stats.txt) contain evaluation results.
The file results.txt contains prediction results for each test sample.
The file stats.txt contains precision / recall / F1 score statistics
for each class category. The rest (model.txt and features.txt) contain 
model definitions; you can use these two files to load a pretrained model. 

Let's check the result file stats.txt:
```
$ cat stats.txt
TESTING DATA SET

Calculations per sequences:                                                                     
Label   True+   Marked+ Detect+ Prec.   Recall  F1                                              
0:      3       3       3       100     100     100                                             
1:      3       3       3       100     100     100                                             
2:      3       3       3       100     100     100                                             
-----------------------------------------------------------------------                         
Ov:     9       9       9       100     100     100   
```
Our test dataset contained 9 samples in total, equally divided into three 
classes. We can see that the trained model correctly predicted all 9 samples.

## Change log  
- ver 3.0 Added One-Class CRF and HCRF (OCCRF, OCHCRF) [5]. Dropped Windows support, switched to LINUX platforms.
- ver 2.0 Added Hierarchical Sequence Summarization (HSS) HCRF [4]. Changed the name of the project [testMVLDM] to [testModels]  
- ver 1.1 Added a project matHCRF, a matlab wrapper  
- ver 1.0 Initial release  
  
## Disclaimer
There is a patent pending on the ideas presented in One-class models (OCCRF and OCHCRF); so this code should only be used for academic purposes only.

## References
[1] Ariadna Quattoni, Sy Bor Wang, Louis-Philippe Morency, Michael Collins, Trevor Darrell: Hidden Conditional Random Fields. TPAMI 2007  
  
[2] Louis-Philippe Morency, Ariadna Quattoni, Trevor Darrell: Latent-Dynamic Discriminative Models for Continuous Gesture Recognition. CVPR 2007  

[3] Yale Song, Louis-Philippe Morency, Randall Davis: Multi-View Latent Variable Discriminative Models for Action Recognition. CVPR 2012  

[4] Yale Song, Louis-Philippe Morency, Randall Davis: Action Recognition by Hierarchical Sequence Summarization. CVPR 2013  

[5] Yale Song, Zhen Wen, Ching-Yung Lin, Randall Davis: One-class Conditional Random Fields for Sequential Anomaly Detection. IJCAI 2013
