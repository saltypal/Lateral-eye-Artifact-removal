"""Before/after plots, preserving signed and absolute correlation meanings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .frontal_benchmark import METHODS
from .frontal_figures import LABELS


def make_figures(output):
    output=Path(output);directory=output/'figures';directory.mkdir(exist_ok=True)
    paths=[]
    def save(figure,name):
        figure.savefig(directory/(name+'.png'),dpi=160,bbox_inches='tight')
        figure.savefig(directory/(name+'.pdf'),bbox_inches='tight')
        paths.append(str(directory/(name+'.png')));plt.close(figure)
    def before_after(frame,before,after,title,xlabel,name):
        figure,axis=plt.subplots(figsize=(11,5))
        for position,method in enumerate(METHODS):
            values=frame.loc[frame.method==method]
            a,b=values[before].mean(),values[after].mean()
            axis.plot([a,b],[position,position],color='.6')
            axis.scatter([a],[position],marker='o',facecolors='none',edgecolors='black',s=60,
                         label='Before' if position==0 else None)
            axis.scatter([b],[position],marker='D',s=40,label='After' if position==0 else None)
        axis.set(yticks=range(6),yticklabels=[LABELS[x] for x in METHODS],xlabel=xlabel,title=title)
        axis.invert_yaxis();axis.grid(axis='x',alpha=.2);axis.legend()
        save(figure,name)
    paired=pd.read_csv(output/'klados_correlation_per_record.csv')
    before_after(paired,'clean_cc_before','clean_cc_after',
        'Klados: correlation with paired clean EEG, before and after','Ordinary Pearson CC (higher is better)',
        '01_klados_clean_cc')
    for axis in ['heog','veog']:
        before_after(paired,axis+'_abs_cc_before',axis+'_abs_cc_after',
            f'Klados frontal: residual {axis.upper()} association','Mean absolute Pearson CC; preservation must also be checked',
            f'02_klados_{axis}')
    native=pd.read_csv(output/'osf_frontal_participant_correlations.csv')
    for condition,axis in [('lateral','heog'),('blink','veog'),('vertical','veog')]:
        subset=native.loc[native.condition==condition]
        before_after(subset,axis+'_abs_cc_before',axis+'_abs_cc_after',
            f'OSF {condition}: frontal {axis.upper()} association, equal participant weight',
            'Mean absolute Pearson CC; smaller association alone does not prove EEG preservation',
            f'03_osf_{condition}_{axis}')
    rest=pd.read_csv(output/'osf_frontal_rest_preservation.csv').groupby(['method','participant'],as_index=False).mean(numeric_only=True)
    for metric,title,unit,name in [
        ('relative_change','OSF resting EEG: input modification','Relative RMS change','04_rest_change'),
        ('resting_input_cc','OSF resting EEG: agreement with original input','Pearson CC; input is not clean ground truth','05_rest_cc'),
        ('covariance_relative_error','OSF resting EEG: spatial covariance change','Relative covariance error','06_rest_covariance')]:
        figure,axis=plt.subplots(figsize=(11,5))
        for position,method in enumerate(METHODS):
            values=rest.loc[rest.method==method,metric]
            axis.scatter(values,np.full(len(values),position),s=28)
            axis.plot(values.mean(),position,'kD')
        axis.set(yticks=range(6),yticklabels=[LABELS[x] for x in METHODS],xlabel=unit,title=title)
        axis.invert_yaxis();axis.grid(axis='x',alpha=.2)
        save(figure,name)
    figure,axes=plt.subplots(1,2,figsize=(14,5))
    for axis,band in zip(axes,['alpha','beta']):
        values=rest.groupby('method')[band+'_error_db'].mean().reindex(METHODS)
        axis.barh(range(6),values)
        axis.set(yticks=range(6),yticklabels=[LABELS[x] for x in METHODS],
                 xlabel='Absolute band-power change (dB)',title='Resting '+band+' preservation proxy')
        axis.invert_yaxis()
    figure.tight_layout();save(figure,'07_rest_spectral')
    channels=pd.read_csv(output/'osf_frontal_before_after_per_channel.csv')
    for condition,reference in [('lateral','heog'),('blink','veog')]:
        values=channels[(channels.condition==condition)&channels.method.isin(METHODS)]
        values=values.pivot_table(index='method',columns='channel',values=reference+'_abs_cc_after').reindex(METHODS)
        figure,axis=plt.subplots(figsize=(11,5))
        picture=axis.imshow(values.values,aspect='auto',vmin=0,vmax=1,cmap='magma')
        axis.set(yticks=range(6),yticklabels=[LABELS[x] for x in METHODS],
                 xticks=range(len(values.columns)),xticklabels=values.columns,
                 title=f'OSF {condition}: after-cleaning absolute {reference.upper()} CC by frontal electrode')
        figure.colorbar(picture,ax=axis);save(figure,f'08_channel_{condition}')
    for condition in ['blink','lateral']:
        chosen=channels[channels.condition==condition].iloc[0]
        filename=output/'osf'/chosen['record']/f"trial_{int(chosen['trial']):03d}_predictions.npz"
        with np.load(filename,allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
        time=np.arange(data['raw'].shape[-1])/200
        figure,axes=plt.subplots(7,1,figsize=(13,14),sharex=True)
        for axis,method in zip(axes[:6],METHODS):
            axis.plot(time,data['raw'][0],color='.7',label='Before F3')
            axis.plot(time,data[method][0],label=LABELS[method],alpha=.8)
            axis.legend(loc='upper right',fontsize=8)
        axes[-1].plot(time,data['heog'],label='Unchanged HEOG')
        axes[-1].plot(time,data['veog'],label='Unchanged VEOG')
        axes[-1].legend();axes[-1].set_xlabel('Seconds')
        figure.suptitle(f'OSF {condition}: actual frontal output and unchanged references; no clean target')
        save(figure,f'09_waveforms_{condition}')
    modes=pd.read_csv(output/'osf_frontal_mode_eog_correlations.csv')
    modes=modes[modes.method=='historical_vmd'].copy()
    modes[['heog_cc','veog_cc']]=modes[['heog_cc','veog_cc']].abs()
    table=modes.groupby(['condition','mode'])[['heog_cc','veog_cc']].mean()
    figure,axis=plt.subplots(figsize=(9,7))
    picture=axis.imshow(table.values,aspect='auto',vmin=0,vmax=1,cmap='magma')
    axis.set(yticks=range(len(table)),yticklabels=[f'{a}, mode {b}' for a,b in table.index],
             xticks=[0,1],xticklabels=['HEOG','VEOG'],title='OSF: every frontal VMD mode checked against both EOG axes')
    figure.colorbar(picture,ax=axis);save(figure,'10_native_mode_correlations')
    failures=json.loads((output/'osf_method_failures.json').read_text())
    figure,axis=plt.subplots(figsize=(11,5))
    axis.barh(range(6),[sum(row['method']==method for row in failures) for method in METHODS])
    axis.set(yticks=range(6),yticklabels=[LABELS[x] for x in METHODS],
             xlabel='Failed windows retained as pass-through',title='Native method failures')
    save(figure,'11_failures')
    return paths


def write_review(output):
    output=Path(output)
    klados=pd.read_csv(output/'klados_correlation_summary.csv').set_index('method')
    osf=pd.read_csv(output/'osf_frontal_correlation_summary.csv')
    complete=json.loads((output/'native_completion.json').read_text())
    text='# Frontal correlation audit and native OSF results\n\n'
    text+='Klados saved predictions were independently correlated against the paired clean EEG. Before and after are explicit, and each rewritten NPZ contains electrode names and source row indices.\n\n'
    text+=klados[['clean_cc_before','clean_cc_after','heog_abs_cc_before','heog_abs_cc_after',
                  'veog_abs_cc_before','veog_abs_cc_after']].round(6).to_markdown()+'\n\n'
    text+='## OSF frontal only\n\n'
    text+=f"Sessions: {complete['sessions']}; participants: {complete['participants']}; scored trials: {complete['trials']}. Shared verified frontal names: {complete['processed_channel_names']}.\n\n"
    text+='FP1/FP2 are absent in the selected OSF montage; FPz is not substituted. All methods receive the same five frontal inputs. EEGOAR-Net has those two original custom-montage inputs marked absent; this missing-context transfer was not trained separately. No posterior or central denoising is performed.\n\n'
    text+=osf[['method','condition','heog_abs_cc_before','heog_abs_cc_after',
               'veog_abs_cc_before','veog_abs_cc_after']].round(6).to_markdown(index=False)+'\n\n'
    text+='Participant-balanced means average trials/channels inside each participant before averaging participants. Signed values and individual channels remain in CSVs. Absolute correlations are computed before averaging so left/right signs cannot cancel. Evaluation H/V references are unchanged.\n\n'
    text+='There is no paired clean EEG target in native OSF. Resting input correlation, modification, PSD and covariance are preservation proxies, not clean reconstruction accuracy or SNR. Neural checkpoints are frozen from Klados; MWF uses separate native calibration trials. Source hashes, exact labels, trial identities, failures and figures are saved. No parameter tuning is performed using these scoring trials. Author-pretrained OSF participant exposure is not independently verified, so this is exploratory development validation.\n'
    (output/'RESULTS.md').write_text(text,encoding='utf-8')
