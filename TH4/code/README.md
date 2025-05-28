# EEG_epilepsy_classification_Bonn_dataset

Epilepsy is one of the world's most common neurological diseases. Early prediction of the incoming seizures has a great influence on epileptic patients' life. If the occurrence of seizure could be predicted well in advance, it could be prevented through medication or proper actions. Electroencephalogram (EEG) is generally used to detect epilepsy as EEG is capable of capturing the electrical activity of brain. In literature, many machine learning techniques are used to predict the occurrence of seizures in EEG recordings. 
In this repository, I try multiple machine learning algorithms to classify randomized epochs of EEG signals into seizure and non-seizure classes. For training and testing, I use EEG dataset provided by Bonn University’s Epileptology department which presents Electroencephalogram (EEG) recordings of 500 individuals containing non-seizure and seizure data. 

I implemented two methods to classify EEG signals into seizure and non-seizure classes. The first method works with the original signal values while the second method decomposes signal into common timesub-bands using discrete wavelet transformation. For each of these methods I use different machine learning algorithms to achieve the best results. 

Please refer to EEG_Classification_Report.pdf for a detailed report of the implemented methods and results.
