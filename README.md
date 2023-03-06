# VocalAdversary2022 # 
## extended results for the 'Beyond Neural-on-Neural Approaches to Speaker Gender Protection' paper. ##

Adversarial protection of gender in speech, tested on both Voxceleb and LibriSpeech.
The 'Reference model' is the model that was used to generate the adversarial speech, while the 'attack model' is the classifier that is attempting to obtain the gender and we want to protect against it. The lower the accuracy, the more successful our adversarial protection is.

Excel sheets contain full results split up into datasets,
**"Voxceleb_accuracyresults.xlsx"** contains results on the Voxceleb2 test set, also listed here.
|                |              |              |           |                 |                 | **Attack**   | **Model**   |              |             |                   |                   |                |
|----------------|--------------|--------------|-----------|-----------------|-----------------|--------------|-------------|--------------|-------------|-------------------|-------------------|----------------|
|                | -> M5-LS100h | -> M5-LS960h | -> M5-vox | -> WavLM-LS100h | -> WavLM-LS960h | -> WavLM-vox | -> SVM-top8 | -> SVM-top10 | -> SVM-full | -> Xvector-LS100h | -> Xvector-LS960h | -> Xvector-vox |
| **Reference Model**  |              |              |           |                 |                 |              |             |              |             |                   |                   |                |
| Normal Voxtest | 92,52        | 95,47        | 97,11     | 97,73           | 97              | 99,02        | 85,17       | 87,59        | 79,7        | 89,44             | 97                | 98,45          |
| M5-LS100h      | 0,2          | 11,67        | 51,1      | 97,8            | 96,69           | 98,93        | 86,2        | 88,1         | 80,93       | 83,8              | 95,3              | 98,41          |
| M5-LS960h      | 14,21        | 0,84         | 31,88     | 97,92           | 96,83           | 98,95        | 85,81       | 88,03        | 82,66       | 87,78             | 95,56             | 98,29          |
| M5-vox         | 59,86        | 48,8         | 0,26      | 97,82           | 97,1            | 98,86        | 86,15       | 88,28        | 82,94       | 89,24             | 95,3              | 97,52          |
| WavLM-LS100h   | 92,76        | 95,37        | 96,98     | 3               | 26,68           | 34,96        | 85,48       | 87,58        | 82,7        | 87,26             | 93,4              | 96,08          |
| WavLM-LS960h   | 92,79        | 95,47        | 95,47     | 47,09           | 3,93            | 57,64        | 85,15       | 87,22        | 81,76       | 90,61             | 94,15             | 96,47          |
| WavLM-vox      | 92,58        | 95,46        | 97,1      | 85,07           | 81,08           | 6,25         | 85,47       | 87,78        | 81,66       | 92,01             | 96,26             | 96,21          |



**"Librispeech_accuracyresults.xlsx"** contains results on the Librispeech test-clean set. 
|                          |              |              |           |                 |                 | **Attack**   | **Model**   |              |             |                   |                   |                |
|--------------------------|--------------|--------------|-----------|-----------------|-----------------|--------------|-------------|--------------|-------------|-------------------|-------------------|----------------|
|                          | -> M5-LS100h | -> M5-LS960h | -> M5-vox | -> WavLM-LS100h | -> WavLM-LS960h | -> WavLM-vox | -> SVM-top8 | -> SVM-top10 | -> SVM-full | -> Xvector-LS100h | -> Xvector-LS960h | -> Xvector-vox |
| **Reference Model**            |              |              |           |                 |                 |              |             |              |             |                   |                   |                |
| Normal   LibriSpeechtest | 96,56        | 96,83        | 92,98     | 97,81           | 98,71           | 98,35        | 89,34       | 94,22        | 92,29       | 96,68             | 97,79             | 97,56          |
| M5-LS100h                | 0            | 8,27         | 48,28     | 97,1            | 97,8            | 98,08        | 89,42       | 89,42        | 86,35       | 90,99             | 96,98             | 96,9           |
| M5-LS960h                | 13,52        | 0            | 29,82     | 97,77           | 98,2            | 97,77        | 90,6        | 90,36        | 87,49       | 92,75             | 97,33             | 97,1           |
| M5-vox                   | 57,37        | 38,68        | 0         | 98,12           | 94,24           | 97,34        | 91,03       | 90,99        | 87,73       | 90,99             | 96,79             | 96,28          |
| WavLM-LS100h             | 96,54        | 96,39        | 94,58     | 1,65            | 16,07           | 20,61        | 89,68       | 89,83        | 86,42       | 87,89             | 96                | 94,93          |
| WavLM-LS960h             | 96,62        | 96,62        | 94,08     | 44,47           | 0,94            | 41,78        | 91,7        | 91,35        | 87,93       | 92,7              | 96,46             | 95,73          |
| WavLM-vox                | 96,62        | 96,77        | 93,5      | 77,39           | 62,77           | 0,39         | 91,27       | 91,23        | 87,93       | 93,58             | 97,46             | 95,08          |

For accuracies split up in gender, please refer to the excel sheets. 

**"VocalAdversary_accuracy.xlsx"** contains results for our own recorded speech adaptations for the vocal adversary, including a brief description of each adaptation.
|                      |              |              |           |                 |                 | **Attack Model** |              |             |                   |                   |                |
|----------------------|--------------|--------------|-----------|-----------------|-----------------|------------------|--------------|-------------|-------------------|-------------------|----------------|
|                      | -> M5-LS100h | -> M5-LS960h | -> M5-vox | -> WavLM-LS100h | -> WavLM-LS960h | -> WavLM-vox     | -> SVM-top10 | -> SVM-full | -> Xvector-LS100h | -> Xvector-LS960h | -> Xvector-vox |
| **Voice adaptation** |              |              |           |                 |                 |                  |              |             |                   |                   |                |
| default              | 100          | 95,83        | 62,5      | 100             | 100             | 100              | 100          | 100         | 100               | 87,5              | 100            |
| whisper              | 50           | 50           | 50        | 93,75           | 93,75           | 97,92            | 50           | 50          | 54,17             | 60,42             | 100            |
| lowrobot             | 100          | 62,5         | 50        | 100             | 100             | 100              | 100          | 100         | 100               | 89,58             | 95,83          |
| highrobot            | 100          | 97,92        | 70,8      | 100             | 100             | 100              | 100          | 100         | 100               | 100               | 100            |
| fan                  | 100          | 58,33        | 58,33     | 97,92           | 100             | 97,92            | 98           | 98          | 52,08             | 54,17             | 100            |
| overlyhappy          | 50           | 52,08        | 62,5      | 56,25           | 70,83           | 66,67            | 50           | 50          | 50                | 60,42             | 85,42          |
| mediumroomreverb     | 100          | 95,74        | 59,57     | 100             | 100             | 100              | 100          | 97,5        | 100               | 82,98             | 100            |
| whisperplus25db      | 72,92        | 66,67        | 79,17     | 89,58           | 75              | 100              | 50           | 50          | 93,75             | 95,83             | 100            |
| stagewhisper         | 73,47        | 100          | 75,51     | 100             | 89,58           | 100              | 50           | 50          | 69,39             | 91,84             | 100            |
| handwhisper          | 64,58        | 50           | 50        | 89,58           | 87,5            | 52,08            | 50           | 50          | 72,92             | 50                | 52,08          |
| hand                 | 100          | 87,5         | 50        | 100             | 100             | 91,67            | 100          | 100         | 100               | 83,33             | 93,75          |
| nose                 | 100          | 95,83        | 75        | 100             | 100             | 100              | 100          | 98          | 100               | 97,92             | 100            |
| cheek                | 100          | 94           | 54        | 100             | 100             | 100              | 96           | 98          | 100               | 92                | 100            |
| penfront             | 100          | 85,41        | 64,17     | 100             | 100             | 95,83            | 100          | 100         | 100               | 85,42             | 91,67          |
| penback              | 100          | 85,41        | 54,17     | 100             | 100             | 89,58            | 98           | 100         | 97,92             | 87,5              | 81,25          |
| towel                | 100          | 97,92        | 60,42     | 100             | 100             | 100              | 96           | 100         | 100               | 75                | 100            |
| cup                  | 97,92        | 95,83        | 62,5      | 97,92           | 100             | 100              | 92,5         | 95,5        | 91,67             | 79,17             | 100            |
| fanlowrobot          | 100          | 56,25        | 54,17     | 100             | 100             | 100              | 100          | 100         | 58,33             | 77,08             | 100            |
| fanhighrobot         | 100          | 85,42        | 75        | 100             | 100             | 100              | 100          | 95,5        | 58,33             | 97,92             | 100            |
| fanstagewhisper      | 100          | 75           | 64,58     | 50              | 62,5            | 58,33            | 60,5         | 73,5        | 58,33             | 95,83             | 87,5           |

**data_utiliy-main** contains code for data utility and **vox2_gender_recognition-master** contains code for the training of the gender classification models, both written by [Nik Vaessen](https://github.com/nikvaessen).

**pgd.py** contains the PGD function used to computationally perturb audio.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Overview of data: 

Dataset       | #speakers (F) | #speakers (M) | #utterances (F)  | #utterances (M) |
------------- | ------------- | ------------- | ---------------- | --------------- |
LS-test-clean | 20            | 39            | 1 389            | 1 231           |
Vox2-test aac | 20            | 79            | 10 711           | 25 526          |

Where F denotes the female speakers and M denotes the male speakers.

Results show the accuracy scores for a binary gender classification for the two different datasets that are computationally perturbed with different models we refer to as the 'Reference models'.
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
