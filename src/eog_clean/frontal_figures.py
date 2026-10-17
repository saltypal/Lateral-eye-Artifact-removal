"""Matched comparison plots; all rendering happens inside Kaggle."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from .frontal_benchmark import METHODS


LABELS = {'historical_vmd':'Historical VMD + EOG model',
          'svmd_fcm_adaptation':'SVMD–FCM adaptation',
          'vme_gmetv_adaptation':'VME–GMETV adaptation',
          'eegoarnet_finetuned':'EEGOAR-Net Klados fine-tuned',
          'gevd_mwf':'GEVD–MWF', 'bandroutenet':'BandRouteNet trained',
          'identity':'Unprocessed input','direct_eog_regression':'Direct EOG regression',
          'resampling_identity':'Resampling only'}


def make_figures(output):
    output = Path(output)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    paths = []
    def save(figure,name):
        figure.savefig(figures/(name+'.png'),dpi=160,bbox_inches='tight')
        figure.savefig(figures/(name+'.pdf'),bbox_inches='tight')
        paths.append(str(figures/(name+'.png')))
        plt.close(figure)
    frame = pd.read_csv(output/'per_record.csv')
    summary = pd.read_csv(output/'comparison.csv').set_index('method')
    methods = ['identity',*METHODS,'direct_eog_regression']
    labels = [LABELS[name] for name in methods]
    for metric,title,unit,name in [
        ('pooled_window_snr_db','Frontal output SNR, equal record weight','dB','01_snr'),
        ('rmse','Frontal per-channel RMSE, equal record weight','source amplitude units','02_rmse'),
        ('rrmse_t','Frontal temporal RRMSE','relative error','03_rrmse'),
        ('pearson_cc','Frontal ordinary Pearson correlation','CC','04_pearson')]:
        figure,axis = plt.subplots(figsize=(10,5))
        for position,method in enumerate(methods):
            values = frame.loc[frame.method==method,metric]
            axis.scatter(values,np.full(len(values),position),s=18,alpha=.5)
            axis.plot(summary.loc[method,metric],position,'kD',markersize=6)
        axis.set(yticks=range(len(methods)),yticklabels=labels,xlabel=unit,title=title)
        axis.invert_yaxis()
        axis.grid(axis='x',alpha=.25)
        figure.text(.01,.005,'Dots: individual records. Black diamond: record mean. Development-exposed split.',fontsize=9)
        save(figure,name)
    channels = pd.read_csv(output/'per_channel_window.csv')
    heat = channels[channels.method.isin(METHODS)].pivot_table(
        index='method',columns='channel',values='snr_out_db')
    heat = heat.reindex(METHODS)
    figure,axis = plt.subplots(figsize=(11,5))
    visual = axis.imshow(heat.values,aspect='auto',cmap='viridis')
    axis.set(xticks=range(len(heat.columns)),xticklabels=heat.columns,
             yticks=range(len(heat)),yticklabels=[LABELS[x] for x in heat.index],
             title='Per-channel output SNR; paper-listed channel order assumption')
    figure.colorbar(visual,ax=axis,label='dB')
    save(figure,'05_channel_snr')
    clean = pd.read_csv(output/'clean_preservation.csv')
    record_clean = clean.groupby(['method','record']).mean(numeric_only=True)
    figure,axes = plt.subplots(1,2,figsize=(14,5))
    for axis,metric,unit,gate in [(axes[0],'relative_change','Clean-input change (%)',1.),
                                 (axes[1],'covariance_relative_error','Relative covariance error',.02)]:
        values = [record_clean.loc[method,metric].max() for method in METHODS]
        if metric=='relative_change':values=np.array(values)*100
        axis.barh(range(len(METHODS)),values)
        axis.axvline(gate,color='red',linestyle='--',label='Engineering gate')
        axis.set(yticks=range(len(METHODS)),yticklabels=[LABELS[x] for x in METHODS],xlabel=unit)
        axis.invert_yaxis();axis.legend()
    figure.suptitle('Worst-record mean clean-input modification; smaller is better')
    figure.tight_layout()
    save(figure,'06_preservation')
    associations = channels[channels.method.isin(METHODS)].copy()
    associations[['heog_cc','veog_cc']] = associations[['heog_cc','veog_cc']].abs()
    assoc = associations.groupby('method')[['heog_cc','veog_cc']].mean().reindex(METHODS)
    figure,axis = plt.subplots(figsize=(11,5))
    visual = axis.imshow(assoc.values,vmin=0,vmax=1,aspect='auto',cmap='magma')
    axis.set(xticks=[0,1],xticklabels=['HEOG','VEOG'],yticks=range(6),
             yticklabels=[LABELS[x] for x in METHODS],title='Residual absolute EOG correlation (not a preservation score)')
    figure.colorbar(visual,ax=axis)
    save(figure,'07_eog_correlations')
    directory = sorted((output/'windows').glob('*'))[0]
    with np.load(directory/'predictions.npz',allow_pickle=False) as archive:
        data = {name:archive[name] for name in archive.files}
    time = np.arange(1024)/200
    figure,axes = plt.subplots(6,1,figsize=(13,14),sharex=True,sharey=True)
    for axis,method in zip(axes,METHODS):
        axis.plot(time,data['raw'][0],color='.8',label='Input')
        axis.plot(time,data['clean'][0],color='black',label='Paired clean target')
        axis.plot(time,data[method][0],alpha=.8,label=LABELS[method])
        axis.legend(loc='upper right',fontsize=8)
    axes[-1].set_xlabel('Seconds')
    figure.suptitle('First scored FP1 window, identical axes; example is not the aggregate result')
    save(figure,'08_waveforms')
    figure,axis = plt.subplots(figsize=(11,5))
    for method in ['clean','raw',*METHODS]:
        frequency,power=welch(data[method][0],fs=200,nperseg=512)
        keep=(frequency>=.5)&(frequency<=40)
        axis.semilogy(frequency[keep],power[keep],label=LABELS.get(method,method))
    axis.set(xlabel='Hz',ylabel='Power / Hz (source units squared)',title='First FP1 window spectral comparison')
    axis.legend(fontsize=8,ncol=2)
    save(figure,'09_psd')
    figure,axes=plt.subplots(1,2,figsize=(13,5))
    for axis,name,label in zip(axes,['bandroute','eegoar'],['BandRouteNet','EEGOAR-Net fine-tuning']):
        history = pd.read_csv(output/f'{name}_training.csv')
        axis.plot(history.epoch,history.train_mse,'o-',label='Training')
        axis.plot(history.epoch,history.validation_mse,'o-',label='Validation')
        axis.set(xlabel='Epoch',ylabel='Input-normalized MSE',title=label+'; validation-only selection')
        axis.legend()
    figure.tight_layout()
    save(figure,'10_training')
    modes=pd.read_csv(output/'mode_reference_correlations.csv')
    hist=modes[modes.method=='historical_vmd'].copy()
    hist[['heog_cc','veog_cc']]=hist[['heog_cc','veog_cc']].abs()
    table=hist.groupby(['channel','mode'])[['heog_cc','veog_cc']].mean()
    figure,axis=plt.subplots(figsize=(8,12))
    visual=axis.imshow(table.values,vmin=0,vmax=1,aspect='auto',cmap='magma')
    axis.set(xticks=[0,1],xticklabels=['HEOG','VEOG'],yticks=range(len(table)),
             yticklabels=[f'{a} mode {b}' for a,b in table.index],title='Every historical VMD mode checked against both EOG references')
    figure.colorbar(visual,ax=axis,label='Mean absolute Pearson CC')
    save(figure,'11_mode_reference_evidence')
    failures=json.loads((output/'failures.json').read_text())
    figure,axis=plt.subplots(figsize=(11,5))
    axis.barh(range(6),[sum(x['method']==method for x in failures) for method in METHODS])
    axis.set(yticks=range(6),yticklabels=[LABELS[x] for x in METHODS],
             xlabel='Failed windows retained as identity fallback',title='All method failures remain in scoring')
    save(figure,'12_failures')
    with np.load(directory/'historical_modes.npz',allow_pickle=False) as archive:
        vectors,residual=archive['modes'][0],archive['residual'][0]
    figure,axes=plt.subplots(6,1,figsize=(12,12),sharex=True)
    axes[0].plot(time,data['raw'][0],label='Original FP1');axes[0].legend(loc='upper right')
    for axis,index in zip(axes[1:4],range(3)):
        axis.plot(time,vectors[index],label=f'VMD mode {index+1}')
        axis.legend(loc='upper right')
    axes[4].plot(time,residual,label='Retained reconstruction residual');axes[4].legend(loc='upper right')
    axes[5].plot(time,data['eyes'][0],label='HEOG');axes[5].plot(time,data['eyes'][1],label='VEOG')
    axes[5].legend(loc='upper right');axes[5].set_xlabel('Seconds')
    figure.suptitle('Actual K3/alpha2000 mode vectors, residual and aligned EOG; independent amplitude axes')
    save(figure,'13_mode_vectors')
    return paths
