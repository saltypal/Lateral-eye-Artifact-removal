# Datasets used in this study

We used three public data sources for complementary purposes: Klados for legacy paired reconstruction experiments, Kobler’s OSF data for real ocular artifacts and named-channel regional analysis, and LEMON for low-ocular EEG references. We also constructed controlled LEMON–OSF mixtures. Published dataset characteristics and the subsets used in our experiments are distinguished below.

## Klados semi simulated EEG EOG dataset

Klados and Bamidis released paired pre-contamination and ocular-contaminated EEG, allowing an algorithm’s output to be compared with the EEG before artificial contamination [1, 2]. The published collection contains 54 recordings from 27 participants, with 19 EEG channels sampled at 200 Hz. This makes it useful for assessing reconstruction error rather than relying only on reduced EEG–EOG association.

In our project, converted arrays containing 53 records, 19 signal rows and 5,401 samples per record were used for legacy comparisons and replay of the historical VMD hybrid. These counts describe the local conversion, not the complete published release. The converted files do not establish participant identities, electrode ordering or physical amplitude units. We therefore cannot assume that a particular row is frontal or claim participant-independent validation from these arrays. Previously reused records are development-exposed.

Dataset: https://data.mendeley.com/datasets/wb6yvr725d/3
Paper: https://doi.org/10.1016/j.dib.2016.06.032
The public deposit above is a source reference; its version is not a verified version identity for our older converted arrays.

## Kobler EEG eye artifact dataset on OSF

The OSF EEG eye artifact dataset contains real EEG and EOG recorded during eye movements, blinks and rest [3, 4]. The publisher readme accompanying our copy describes four studies, 45 recording sessions and 39 participants; participant IDs are unique across studies. Sessions and participants must therefore be counted separately. EEGLAB SET/FDT files retain channel labels, trial labels, sample-level artifact annotations and experimental-block information.

Trial conditions are rest, horizontal eye movement, vertical eye movement and blink. Sample annotations further distinguish right, left, up and down movements, blink and rest. Horizontal EOG (HEOG) provides lateral-movement evidence; vertical EOG (VEOG) provides blink/vertical-movement evidence. Verified electrode names support anatomical frontal, posterior and central/shared assignments.

We used OSF data for ocular donors, unscored calibration and native suppression/preservation evaluation. Our development inventory contains 36 recordings. The latest frozen frontal pilot evaluated all 51 scoring trials from study01_p02_prep and study01_p03_prep. Native recordings have no paired artifact-free EEG: EEG–EOG correlation, resting RMSE and spectral preservation are appropriate here; reconstruction SNR is not. Native physical amplitude units remain unverified in our pipeline.

Dataset: https://osf.io/2qgrd/
Dataset DOI: https://doi.org/10.17605/OSF.IO/2QGRD
Author implementation: https://github.com/rkobler/eyeartifactcorrection

The OSF collection was also used in EEGOAR-Net [6]. EEGOAR-Net is a method and evaluation reference, not the creator of this dataset.

## LEMON EEG data

The Leipzig Study for Mind–Body–Emotion Interactions (LEMON) provides multimodal data from 227 participants; its publication describes a 62-channel resting-state EEG experiment [5]. We used an eligible development subset of raw EEG from 32 identified recipients, rather than the entire published cohort. Participant identifiers and named electrodes support source-level separation and montage matching.

Selected low-ocular EEG segments serve as recipient/reference signals for controlled contamination. Their role is a reconstruction reference, not proof that every sample is biologically artifact-free. MNE-loaded EEG retains volts in this processing path. LEMON’s original acquisition and our common 200 Hz processing configuration are separate specifications.

EEG download page: http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/downloads/download_EEG.html
Publisher-linked archive: https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/
Paper: https://doi.org/10.1038/sdata.2018.308

## Controlled LEMON OSF corpus constructed for this project

We combined LEMON recipient EEG with ocular fields derived from OSF donor calibration. Joint HEOG/VEOG regression estimates the donor field, which is mapped to verified common EEG channels and added to recipient EEG. The recipient signal before addition is retained as the paired reference. The repaired corpus preserves the declared coupling between the added field and its EOG reference scaling.

The broader development comparison contains 320 examples from 32 recipients and five ocular donors: ten variants per pair, comprising an idealized clean control and blink, lateral and mixed contamination at nominal whole-montage input SNRs of −5, 0 and +5 dB. These are variants of one 5.12-second recipient window per participant, not 320 independent recordings. Recipient and donor identities remain recorded in the frozen grouped partitions.

This corpus enables paired SNR, MSE, temporal RRMSE and Pearson correlation evaluation. Its regression-generated fields can favor reference-regression methods, and zero-driver clean controls simplify preservation. Native and independent validation are therefore necessary. Earlier corpus versions remain separate evidence; their results must not be pooled silently with the repaired version. This is a project-derived corpus with upstream citations [3–5], not a separately published dataset with its own DOI.

## Common handling and evaluation scope

The main campaign uses 200 Hz sampling, a 0.5–40 Hz band, 1,024-sample windows and a 512-sample hop. Trials and calibration/scoring segments are processed separately; cached preprocessed trials are not filtered again. Confirmation sources remain closed during development selection. EEGdenoiseNet, Magdeburg and other proposed external sources are not counted here as completed waveform evaluations.

## References

[1] Klados, M. A., and Bamidis, P. D. (2016). A semi-simulated EEG/EOG dataset for the comparison of EOG artifact rejection techniques. Data in Brief, 8, 1004–1006. https://doi.org/10.1016/j.dib.2016.06.032

[2] Klados, M., and Bamidis, P. D. (2016). A semi-simulated EEG/EOG dataset for the comparison of EOG artifact rejection techniques. Mendeley Data, version 3. https://doi.org/10.17632/wb6yvr725d.3

[3] Kobler, R. J., Sburlea, A. I., Lopes-Dias, C., Schwarz, A., Mondini, V., and Müller-Putz, G. R. EEG eye artifact dataset. Open Science Framework. https://doi.org/10.17605/OSF.IO/2QGRD. Dataset structure and session counts follow the publisher readme distributed with our copy; accessed 16 October 2026.

[4] Kobler, R. J., et al. (2020). Corneo-retinal-dipole and eyelid-related eye artifacts can be corrected offline and online in electroencephalographic and magnetoencephalographic signals. NeuroImage, 218, 117000. https://doi.org/10.1016/j.neuroimage.2020.117000

[5] Babayan, A., et al. (2019). A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. Scientific Data, 6, 180308. https://doi.org/10.1038/sdata.2018.308

[6] Marcos-Martínez, D., et al. (2025). Calibration-free ocular artifact reduction in EEG signals using a montage-independent deep learning model. Biomedical Signal Processing and Control, 110, 108147. https://doi.org/10.1016/j.bspc.2025.108147
