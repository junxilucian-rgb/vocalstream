"""Export standalone scientific figures without modifying the ML environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(PROJECT/'.plotting_cache')
sys.path.insert(0,str(PROJECT/'.plotting_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); args=p.parse_args()
    root=args.run.resolve(); data=json.loads((root/'analysis.json').read_text())
    state=json.loads((root/'run.json').read_text()); s=data['summaries']; timing=data['timing_repeats']
    figures=root/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.2,'savefig.facecolor':'white','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.2),layout='constrained')
    palette={'openunmix_hq':'#697A88','bs_fp32':'#D75A43','bs_amp':'#087F8C','bs_2s':'#845DA3'}
    ax=axes[0,0]
    labels={'openunmix_hq':'Open-Unmix','bs_fp32':'BS FP32 (11 s)','bs_amp':'BS mixed (11 s)','bs_2s':'BS FP32 (2 s)'}
    offsets={'openunmix_hq':(12,8),'bs_fp32':(-86,-23),'bs_amp':(-35,13),'bs_2s':(-36,-28)}
    for key,label in labels.items():
        q=s['expanded_'+key]; ax.scatter(q['rtf'],q['si_sdr_median'],s=70,color=palette[key],zorder=4)
        ax.annotate(label,(q['rtf'],q['si_sdr_median']),xytext=offsets[key],textcoords='offset points',fontsize=9)
    ax.set(xlabel='Processing call RTF (lower is faster)',ylabel='Median vocal SI-SDR (dB)',title='A  Quality and call cost: 14 clips')
    ax.set_xlim(left=0,right=max(s['expanded_'+k]['rtf'] for k in labels)*1.18)
    values=[s['expanded_'+k]['si_sdr_median'] for k in labels]; ax.set_ylim(min(values)-1.5,max(values)+1.7)
    ax=axes[0,1]
    for key in ('bs_fp32','bs_amp'):
        vals=timing[key]['repeat_rtfs']; ax.plot(range(1,len(vals)+1),vals,'o-',color=palette[key],label=labels[key],linewidth=2)
    ax.set(xlabel='Independent run',ylabel='Processing call RTF',title='B  Repeat timing: 3 fixed songs / 90 s')
    ax.set_xticks([1,2,3]); ax.set_ylim(bottom=0); ax.legend(frameon=False)
    ax=axes[1,0]; rng=np.random.default_rng(2026); distributions=[]
    for index,key in enumerate(('openunmix_hq','bs_amp')):
        report=json.loads(Path(state['jobs']['clean_vocal_'+key]['artifact']).read_text())
        vals=[r['quality']['normalized_waveform_rmse'] for r in report['tracks']]
        distributions.append(vals); ax.scatter(index+1+rng.uniform(-.08,.08,len(vals)),vals,alpha=.65,color=palette[key],s=20,zorder=4)
    boxes=ax.boxplot(distributions,positions=[1,2],widths=.4,patch_artist=True,showfliers=False,
                     medianprops={'color':'#18354D','linewidth':2})
    for patch,color in zip(boxes['boxes'],[palette['openunmix_hq'],palette['bs_amp']]): patch.set_facecolor(color); patch.set_alpha(.25)
    ax.set_xticks([1,2],['Open-Unmix','BS mixed']); ax.set(ylabel='Waveform NRMSE (lower preserves more)',title='C  Original-vocal identity input: 14 clips'); ax.set_ylim(bottom=0)
    ax=axes[1,1]; gains=[-30,-20,-10]
    input_scores=[data['robustness'][f'bleed_minus{abs(g)}_bs_amp']['input_si_sdr_median'] for g in gains]
    ax.plot(gains,input_scores,'o--',color='#A4AFB7',label='Unprocessed input',linewidth=2)
    for key in ('openunmix_hq','bs_amp'):
        vals=[data['robustness'][f'bleed_minus{abs(g)}_'+key]['si_sdr_median'] for g in gains]
        ax.plot(gains,vals,'o-',color=palette[key],label=labels[key],linewidth=2)
    ax.set(xlabel='Original backing amplitude gain (dB)',ylabel='Median vocal SI-SDR (dB)',title='D  Controlled backing bleed: 14 clips')
    ax.set_xticks(gains); ax.legend(frameon=False,fontsize=9)
    fig.suptitle('VocalStream: fixed 60-90 s development clips on RTX 4060 Laptop GPU',fontsize=14,color='#18354D')
    fig.savefig(figures/'refinement_overview.png',dpi=180)
    fig.savefig(figures/'refinement_overview.svg')
    plt.close(fig)
    metadata={'matplotlib':matplotlib.__version__,'numpy':np.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'analysis_sha256':hashlib.sha256((root/'analysis.json').read_bytes()).hexdigest()}
    (figures/'plot_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(figures/'refinement_overview.png')


if __name__=='__main__': main()
