MICSR (MICSR IS CZECH SPEECH RECOGNISER)

The main run script is in MICSR.sh, it contains full data preparation (including rewriting text to digits and other preprocessing things). Currently the setup is made in a way that it normalizes each utterance individually.
Theere is also option to perform data augmentation by adding a RIR noises that should make model more robust in large room conditions.
The lexicon consisns od 2.3 M words and LM was trained by interpolation of 65M LM and domain specific ~1M words on a dev set. There is also ready pipeline for segmenting long utterance audios


In DNN we predict from input x output y which is one of the HMM states (e.g. there are 5000 of them). But we do not care only about that one true values but also about the log costs of all the values as they will be used in Viterbi decoding.
Each iteration of DNN training corresponds to processing K examples by all machines. It than averages the parameters and redistributes them across machines. This is also the reason why to start with small number of initial jobs and larger number of final jobs (parameters are quite stable by then).
There is some LR rescaling which results in modification of SGD to natural SGD. The modification is to use Fisher matrix which is kinda approximation to Hessian so the hwole thing works a bit like second order method.


The new model with adapted LM and dictionary is in /lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments/new/graph



nnet3
it has two main building blocks 1) compotents - things we want in our NN 2) how to glue these components together (so for example it supports RNNs)



UBM (https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-73003-5_197)
- universal background model is speaker independent GMM to represent general speech characteristics. We may be interested in e.g. if given audio sample belongs to given person -> compute likelihodd ration test for hypothesis H0 belong H1 does not. We collect person dependent information X to compute this likelihood.
It is quite straight forward to get the speaker specific features as we already have a sample but how to construct alternative hypothesis model that should cover all of the speaker space of all x's. We use large number of samples from different speakers to train one universal model to represent the alternative hypothesis (it is a large mixture of Gaussians representing all speakers).

JFA (http://www1.icsi.berkeley.edu/Speech/presentations/AFRL_ICSI_visit2_JFA_tutorial_icsitalk.pdf)
We have one UBM and want to train speaker supervectors. We do it by fixing all speakers and moving UBM only in direction to one speaker so that is learn that shift. We proceed with all speakers like that. The JFA decomposes the supervector into speaker independent part (from UBM), speaker dependent part, channel dependent part and speaker dependent residual so it is: s = m + Vy + Ux + Dz.
So we represent each speaker with his/her own mixture of Gaussians. Each of speaker and channel dependent matrices can be represented in a low dimmensional space along the principal axes (eigen values?) - so they are y and x.
For test audio we first extract MFCC's and use likelihood to calculate score for all speakers against the UBM and take assign the most likely speaker.

I-vectors
It is all important information about speaker and all kinds of other variablity.
Similar to JFA but combines intra and inter domain variability and model it in some low dimmensional space. It was observed that in JFA the channel dependent factor also models some of the speaker festures so I-vectors are attempt to unite these factorizations to benefit from all relevant information.
New total variablity space T was introduced: M = m + T*x, where m is UBM speaker and channel independent supervector, x in normally distributed random vector in T space and T columns of T represent first n principal components of ????. We assume that utterances produced by one speaker come from different speaker (unlike in JFA). After training x in the I-vector. I-vectors are generative model so we do not need to have data labels.
I-vectors are appended to acoustic features during DNN training and serve as normalizatio
Sufficient statistics is process where sequence of MFCC's is represented by Baum-Welsch statistics obtained from UBM GMM model -> these statistics are high dimmensional and are converted to lower dimmension space to i-vectors.
Senones are set of triphones close together in the acoustic space -> they are determined by DT using ML approach. DT asks locally optimal questions in each split that give the highest rise to the likelihood. Viterbi decoder aligns the data to the senones (slignment step in Kaldi). The alignemnt are used to estimate probability distribution p(x|q), where x is observation and q is senone. Traditionally this distribution was modeled by GMM
We can also think of i-vectors as "substracting the speaker info" and leaving just the information needed for correct senone classification. Imagine input to DNN, which is 39 MFCC and 100 i-vector -> to neuron will go f(x1w1+....+x139w139) -> w's are real numbers so we can imagine having the 100 dim weights with negative sign.


X-vectors
Traditionally for speaker verification UBM model was trained to extract sufficient statistics -> extract I-vectors and finally score them by PLDA to compute similarities between vectors.


Lexicon extension
https://kaldi-asr.org/doc/online_decoding.html
Postup: In decode_with_domain.sh first update the model lexicon, than train LM and subset it against this new lexicon -> update graph -> make segmentations with this new model of Poslanecká sněmovna (make sure that you update lexicon by corpus from it so that you do not have many OOV's)

TODO:
- experiment with normalization (across whole shows not just individual utterances)
- use Switchboard setup parameters for HMM-GMM (they )
- look at Tedlium setup we can decode the data after cleaning to see the results
- LSTM layers (in https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/local/chain/tuning/run_tdnn_lstm_1a.sh - Dan suggests for large lot of data do double the layer sizes from 512 to 1024), take zkusi Oplatkovu architekturu
- Try transfer lerning with already trained chain model (have a look at egs/rm/s5/local/chain/tuning/run_tdnn_wsj_rm_1c.sh). Audio and text is from target is processed using the source model
- some really cool LM text sources: http://opus.nlpl.eu/ (Here are many EU related texts, find out how to work with these), https://www.clarin.eu/resource-families/parallel-corpora
- After getting the data from poslanecká sněmovna try running diarization on top of it and if results are reasonable include in to spk2utt utt2spk when training with it so that may help with accuracy?

- pry nejlepsi online receipt je v /home/jkratochvil/personal_work_ms/00/kaldi/egs/tedlium/s5_r2/local/chain/tuning/run_tdnn_lstm_1e.sh


!!!! Vymyslet to, že když bude někdo mluvit třeba 5 minut tak nahrávat -> vzít audio, udělat transfer learning učení na doladění a "overfitnutí" na toho řečníka a pak napojit no model do rozpoznávače


Thesis structure

1) Intro
- dexcribe the Elitr project

2) Theory
- mainly focus on the nnet3 I-vectors, etc.
- also include X-vectors when you will describe diarization
- good vizualizations of audio are in Praat

2b) Describe Kaldi Toolkit

3) Diarization task

4) Domain adaptation - LM with domain texts, lexicon, data
- augmentation experiments

5) Describe the segmentation pipeline (both Aeneas and with kaldi) and how to iterate to get the data.
- experiment with parameters in that sript (1000 words, 50%...)
- Publish corpus a Poslanecná sněmovna data (maybe largest speech corpus commercialy available?)

6) Real time decoding experiments

7) Real time adaptation (use transfer learning on a short segments of speech)
- try starting here: https://groups.google.com/forum/#!topic/kaldi-help/5V38wFPlyyk and here https://www.danielpovey.com/files/2017_asru_transfer_learning.pdf

Where to start after holidays
1) Preparation of data from Poslanecká Sněmovna, check if text end and beginning fit with audio - put all data together and upload them to cluster -> run segmentation on them
2) Prepare test sets -> from European Parliament cca 2h recognise and than correct it, From Poslanecká Sněmovna alsi about 2 hours do the same, from Česky rozhlas also


Ideas for Poslanecká sněmovna - text file and segment file are the same




Subword extensions to lexicon - good tool is Morfesor
