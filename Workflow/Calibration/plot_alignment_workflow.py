#!/usr/bin/env python3
"""Export a publication-friendly diagram of the camera alignment equations."""
from pathlib import Path
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dasc_alignment_figure_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    root = Path(__file__).resolve().parents[2]
    output = root/'diagnostics/figures'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(14, 24), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    navy, blue, muted, teal = '#172b43', '#2584c6', '#4d6277', '#087f80'
    ax.text(.06, .978, 'Vicon ↔ 3DGS Camera Alignment', fontsize=28, weight='bold',
            color=navy, va='top')
    ax.text(.06, .951, 'From COLMAP camera recovery to a saved coordinate transformation',
            fontsize=13, color=muted, va='top')
    legend = FancyBboxPatch((.06, .895), .88, .042, boxstyle='round,pad=0.008,rounding_size=0.008',
                            facecolor='#edf5fb', edgecolor='none')
    ax.add_patch(legend)
    ax.text(.078, .923, 'FRAMES   G: original 3DGS world    V: Vicon world    B: tracked marker body    C: RGB optical camera',
            fontsize=11.2, color=navy, va='center')
    ax.text(.078, .905, r'$R_{AB}$ maps vectors from B into A.   $c$: camera centre.   $b$: marker-body position.   $s$: metres / GS unit.',
            fontsize=11.2, color=navy, va='center')

    steps = [
        ('Recover the camera trajectory from COLMAP',
         'Read the world-to-camera pose for each image from COLMAP images.bin.',
         r'$p_C = R_{CG,k}\,p_G + t_{CG,k}$',
         r'$c_{G,k}=-R_{CG,k}^{T}t_{CG,k},\qquad R_{GC,k}=R_{CG,k}^{T}$'),
        ('Match image times to Vicon poses',
         'Use sensor-header timestamps; interpolate position linearly and orientation with SLERP.',
         r'$\tau_k=t_k^{\mathrm{RGB}}+\Delta t$',
         r'$b_{V,k}=\mathrm{Interp}_{p}(\tau_k),\qquad R_{VB,k}=\mathrm{SLERP}_{R}(\tau_k)$'),
        ('Convert the marker-body pose to the camera pose',
         r'Fixed mount: $d_B$ is the camera origin in B; $R_{BC}$ maps camera axes into B.',
         r'$c_{V,k}=b_{V,k}+R_{VB,k}\,d_B$',
         r'$R_{VC,k}=R_{VB,k}\,R_{BC}$'),
        ('Fit the scene scale, rotation and translation',
         r'Fit $\Theta=(s,R,t,R_{BC},d_B,\Delta t)$; fix mounting/delay parameters if independently known.',
         r'$c_{V,k}\simeq sR\,c_{G,k}+t,\qquad R_{VC,k}\simeq R\,R_{GC,k}$',
         r'$\min_{\Theta}\ \frac{1}{2}\sum_{k\in\mathcal{F}}\sum_{i=1}^{3}'
         r'\left[\rho\!\left(\frac{e_{p,k,i}^{2}}{\sigma_p^2}\right)'
         r'+\rho\!\left(\frac{e_{R,k,i}^{2}}{\sigma_R^2}\right)\right]$'),
        ('Validate the alignment on excluded poses',
         'Fit on selected time blocks; evaluate excluded blocks and check independent physical dimensions.',
         r'$\mathbf{e}_{p,k}=sR\,c_{G,k}+t-c_{V,k},\qquad E_{p,k}=\|\mathbf{e}_{p,k}\|_2$',
         r'$\mathbf{e}_{R,k}=\mathrm{Log}\!\left(R_{VC,k}^{T}R\,R_{GC,k}\right)^{\vee},'
         r'\qquad E_{\theta,k}=\frac{180}{\pi}\|\mathbf{e}_{R,k}\|_2$'),
        ('Save the calibration; use its inverse for rendering',
         r'Store $s,R,t,R_{BC},d_B,\Delta t$.  Here $R=R_{VG}$.  The original PLY can stay unchanged.',
         r'$c_G=\frac{1}{s}R^{T}(c_V-t)'
         r'=\frac{1}{s}R^{T}\!\left(b_V+R_{VB}d_B-t\right)$',
         r'$R_{GC}=R^{T}R_{VC}=R^{T}R_{VB}R_{BC}$'),
    ]
    for index, (title, note, first, second) in enumerate(steps):
        top = .87-index*.135
        color = teal if index == 5 else blue
        ax.add_patch(FancyBboxPatch((.06, top-.12), .88, .12,
            boxstyle='round,pad=0.002,rounding_size=0.009',
            facecolor='#f1faf8' if index == 5 else '#fafcfe', edgecolor='#baded8' if index == 5 else '#d5e1eb',
            linewidth=1.2))
        ax.text(.079, top-.018, f'{index+1:02d}', fontsize=16, weight='bold', color=color, va='center')
        ax.text(.125, top-.018, title, fontsize=16, weight='bold', color=navy, va='center')
        ax.text(.079, top-.040, note, fontsize=10.9, color=muted, va='center')
        ax.text(.09, top-.069, first, fontsize=19 if index != 4 else 18, color=navy, va='center')
        ax.text(.09, top-.101, second, fontsize=18 if index != 3 else 17.5, color=navy, va='center')
        if index < 5:
            ax.add_patch(FancyArrowPatch((.5, top-.123), (.5, top-.132), arrowstyle='-|>',
                                         mutation_scale=14, linewidth=1.2, color='#7893ac'))

    ax.text(.06, .052,
        r'FIT DETAILS   $\mathcal{F}$: fitting poses; $\rho$: soft-L1 loss; '
        r'$\sigma_p=0.02$ m; $\sigma_R=1^{\circ}$ (converted to radians).',
        fontsize=10.8, color=muted, va='center')
    ax.text(.06, .039, r'$\mathrm{Log}(\cdot)^{\vee}$ converts a relative rotation matrix to a rotation vector in radians.',
            fontsize=10.6, color=muted, va='center')
    ax.text(.06, .023, 'Current calibration: mount and alignment were estimated jointly. Independent mounting verification is still needed.',
            fontsize=10.7, color=teal, va='center')
    for extension in ('png', 'pdf', 'svg'):
        path = output/f'vicon_3dgs_alignment_equations.{extension}'
        fig.savefig(path, dpi=200, facecolor='white', metadata={'Title': 'Vicon to 3DGS Camera Alignment'})
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
