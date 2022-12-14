# VocalAdversary2022 # 
## extended results for the 'Beyond Neural-on-Neural Approaches to Speaker Gender Protection' paper. ##

Excel sheets contain results split up into datasets,
**"Voxceleb_accuracyresults.xlsx"** contains results on the Voxceleb2 test aac set, while **"Librispeech_accuracyresults.xlsx"** contains results on the Librispeech test-clean set. 
**"VocalAdversary_accuracy.xlsx"** contains results for our own recorded speech adaptations for the vocal adversary, including a brief description of each adaptation.

**data_utiliy-main** contains code for data utility and **vox2_gender_recognition-master** contains code for the training of the gender classification models, both written by [Nik Vaessen](https://github.com/nikvaessen).

**pgd.py** contains the PGD function used to computationally perturb audio.


Overview of data: 

Dataset       | #speakers (F) | #speakers (M) | #utterances (F)  | #utterances (M) |
------------- | ------------- | ------------- | ---------------- | --------------- |
LS-test-clean | 20            | 39            | 1 389            | 1 231           |
Vox2-test aac | 20            | 79            | 10 711           | 25 526          |

Where F denotes the female speakers and M denotes the male speakers.

Results show the accuracy scores for a binary gender classification for the two different datasets that are computationally perturbed with different models.
The models used to test the data:

Name           | Also used to perturb the data? | Input to the model   | 
-------------- | ------------------------------ | -------------------- |
M5-LS100h      | Y                              | raw waveform         |
M5-LS960h      | Y                              | raw waveform         |
M5-vox         | Y                              | raw waveform         |
WavLM-LS100h   | Y                              | raw waveform         |
WavLM-LS960h   | Y                              | raw waveform         |
WavLM-vox      | Y                              | raw waveform         |
SVM-top10      | N                              | 10 speech features   |
SVM-full       | N                              | 35 speech features   |
XVector-LS100h | N                              | 40 MFCC coefficients |
XVector-LS960h | N                              | 40 MFCC coefficients |
Xvector-vox    | N                              | 40 MFCC coefficients |

Where LS-100h is the Librispeech-train-clean-100 set, LS-960h is the Librispeech-train-clean-100 + Librispeech-train-clean-360 + Librispeech-train-other-500, and vox is vox2-dev.
[M5](https://ieeexplore.ieee.org/document/7952190/) is a CNN, WavLM is [HuggingFace's transformer](https://huggingface.co/docs/transformers/model_doc/wavlm) and [XVector](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) is known from the Speaker Recognition domain.

The SVM used has a linear kernel and is trained on LS-100h. Speech features used with the SVM are extracted from audio with [Praat](https://www.fon.hum.uva.nl/praat/).
<details>
<summary>The total 35 speech features are</summary>
<p>the number of Pulses, Periods and Voicebreaks; the degree of Voicebreaks, the fraction of Unvoiced parts, jitter (local, local absolute, rap, ppq5), shimmer (local, local dB, apq3, apq5, apq11), mean of the autocorrelation, Noise-to-Harmonics-Ratio (NHR), Harmonics-to-Noise-Ratio (HNR), mean and standard deviation of period and the min, max, mean, median and standard deviation of pitch, as well as the duration, intensity (min, max, mean, standard deviation), the fundamental frequency F0, first three formants and the centre of gravity.</p>
</details>

The 'top10' features are determined with an [SVM-RFE feature selection](https://link.springer.com/article/10.1023/A:1012487302797) to rank features based on importance on the gender classification task with LS-100h.
<details>
<summary>These top 10 features are</summary>
<p>mean, max and std of pitch, mean of autocorrelation, mean of NHR, mean and max of intensity, apq11 and apq3 of shimmer, and local absolute jitter.</p>
</details>

The computationally perturbed data is either LS-test or Vox2-test perturbed with the 'Reference model' listed in the sheet.
For perturbations, [PGD](https://arxiv.org/abs/1706.06083) has been used with cross-entropy loss as loss function, a clipping of 0.1, a perturbation rate of 0.0005 and 10 epochs (for WavLM models) or 100 epochs (for M5 models).


For more information, please refer to our paper "Beyond Neural-on-Neural Approaches to Speaker Gender Protection" or feel free to contact [Loes](mailto:loes.vanbemmel@ru.nl) for questions.
