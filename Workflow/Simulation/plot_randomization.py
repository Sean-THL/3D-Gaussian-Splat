#!/usr/bin/env python3
"""White-background overview of actual seeded domain-randomization samples.

Plots reference trajectories/settings, not MPC tracking or trained-policy results.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dasc-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Workflow.Simulation.randomization import sample_episode
from Workflow.Simulation.tasks.trajectory import make_trajectory

INK = '#19334c'
MUTED = '#64788a'
GRID = '#e6edf2'
COLORS = ['#2979bf', '#119c87', '#db8c35']


def clean_axis(ax):
    ax.set_facecolor('white')
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['bottom', 'left']:
        ax.spines[side].set_color('#c5d0da')
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=.7)
    ax.set_axisbelow(True)


def panel_title(ax, letter, title):
    ax.set_title(f'{letter}   {title}', loc='left', color=INK, fontsize=13, fontweight='bold', pad=16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'Workflow/Simulation/configs/quadrotor_nmpc.json')
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT/'diagnostics/figures/domain_randomization_overview')
    args = parser.parse_args()
    if args.episodes < 3 or args.seed < 0:
        parser.error('Use at least 3 episodes and a nonnegative seed')
    config = json.loads(args.config.read_text())
    if config['dynamics_model'] != 'quadrotor' or config['task']['trajectory_type'] != 'lissajous_figure8':
        parser.error('This overview targets the quadrotor figure-eight profile')
    nominal = make_trajectory(config['task'])
    sampling = copy.deepcopy(config)
    sampling['randomization']['enabled'] = True
    episodes = []
    for seed in range(args.seed, args.seed+args.episodes):
        task, plant, metadata = sample_episode(sampling, np.random.default_rng(seed))
        episodes.append(dict(seed=seed, task=task, plant=plant, randomization=metadata))
    selected = list(dict.fromkeys([0, int(np.argmax([e['task']['duration_s'] for e in episodes])),
                                  int(np.argmin([e['task']['yaw_deg'] for e in episodes]))]))
    for index in range(len(episodes)):
        if len(selected) >= 3:break
        if index not in selected:selected.append(index)
    paths = [make_trajectory(e['task']) for e in episodes]

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'text.color':INK,
        'axes.labelcolor':MUTED, 'figure.facecolor':'white', 'savefig.facecolor':'white',
        'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.unicode_minus':True})
    fig = plt.figure(figsize=(17, 12), facecolor='white')
    grid = fig.add_gridspec(2, 2, left=.065, right=.96, top=.835, bottom=.105,
                           width_ratios=[1.05, 1.0], height_ratios=[1.12, 1.0], hspace=.40, wspace=.27)
    fig.text(.065,.963,'Domain randomization',fontsize=27,fontweight='bold',color=INK)
    fig.text(.065,.929,'One figure-eight task, many starting conditions and drone responses',fontsize=13,color=MUTED)
    fig.text(.96,.963,'QUADROTOR + NMPC',ha='right',fontsize=10,fontweight='bold',color=COLORS[1])
    fig.text(.96,.938,f'{len(episodes)} sampled episodes  |  seeds {args.seed}–{args.seed+len(episodes)-1}',ha='right',fontsize=10,color=MUTED)
    legend = [Line2D([0],[0],color=INK,lw=2,ls='--',label='Nominal reference / model'),
              Line2D([0],[0],color='#acbdca',lw=1.5,label='Other sampled episodes')]
    for index,color in zip(selected,COLORS):
        legend.append(Line2D([0],[0],color=color,lw=2.5,label=f"Seed {episodes[index]['seed']}"))
    fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.058,.902),ncol=5,frameon=False,
               fontsize=10,handlelength=2.5,columnspacing=2.)

    # A: genuine translated reference paths in Vicon coordinates.
    ax = fig.add_subplot(grid[0,0],projection='3d')
    panel_title(ax,'A','Reference paths in Vicon coordinates')
    for route in paths:
        p=np.array(route.polyline(280))
        ax.plot(*p.T,color='#b9c8d3',lw=.9,alpha=.27)
    p=np.array(nominal.polyline(350));ax.plot(*p.T,color=INK,lw=2.1,ls='--',zorder=8)
    for index,color in zip(selected,COLORS):
        route=paths[index];p=np.array(route.polyline(350))
        ax.plot(*p.T,color=color,lw=2.,alpha=.95)
        t=.23*route.duration;position=route.sample(t)[0];heading=route.yaw_reference(t)[0]
        ax.quiver(*position,np.cos(heading),np.sin(heading),0.,length=.11,normalize=True,
                  color=color,linewidth=1.6,arrow_length_ratio=.28)
        start=route.sample(0.)[0]
        perturbed=start+np.asarray(episodes[index]['randomization']['initial_position_offset_m'])
        ax.plot(*np.vstack([start,perturbed]).T,color=color,lw=1.1,ls=':')
        ax.scatter(*perturbed,s=27,color=color,edgecolors='white',linewidths=.7,depthshade=False)
    ax.set_xlabel('X / m',labelpad=9);ax.set_ylabel('Y / m',labelpad=9);ax.set_zlabel('Z / m',labelpad=8)
    ax.set_box_aspect([.9,.85,.36]);ax.view_init(elev=24,azim=-56)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.set_major_locator(MaxNLocator(4));axis.set_pane_color((1,1,1,1))
        axis._axinfo['grid'].update(color=GRID,linewidth=.65)
        axis.line.set_color('#c5d0da')
    ax.tick_params(colors=MUTED,labelsize=8,pad=1)
    ax.text2D(.0,-.06,'Arrows: desired heading at 23% of each cycle.  Dots: perturbed initial positions.',
              transform=ax.transAxes,fontsize=8.5,color=MUTED)

    # B: yaw is not used to rotate the position path; duration changes the clock.
    sub = grid[0,1].subgridspec(2,1,height_ratios=[2.0,1.0],hspace=.35)
    yaw_ax=fig.add_subplot(sub[0]);alt_ax=fig.add_subplot(sub[1],sharex=yaw_ax)
    panel_title(yaw_ax,'B','Heading, altitude and traversal time')
    for axis in [yaw_ax,alt_ax]:clean_axis(axis)
    nominal_t=np.linspace(0,nominal.duration,350)
    yaw_ax.plot(nominal_t,[np.rad2deg(nominal.yaw_reference(t)[0]) for t in nominal_t],color=INK,ls='--',lw=1.8)
    alt_ax.plot(nominal_t,[nominal.sample(t)[0][2] for t in nominal_t],color=INK,ls='--',lw=1.8)
    for index,color in zip(selected,COLORS):
        route=paths[index];times=np.linspace(0,route.duration,350)
        yaw=np.rad2deg([route.yaw_reference(t)[0] for t in times])
        altitude=np.array([route.sample(t)[0][2] for t in times])
        yaw_ax.plot(times,yaw,color=color,lw=2.)
        yaw_ax.scatter(times[-1],yaw[-1],s=24,color=color,zorder=4)
        alt_ax.plot(times,altitude,color=color,lw=2.)
        alt_ax.scatter(times[-1],altitude[-1],s=24,color=color,zorder=4)
    yaw_ax.set_ylabel('Unwrapped yaw / deg');yaw_ax.tick_params(labelbottom=False)
    alt_ax.set_ylabel('Altitude / m');alt_ax.set_xlabel('Simulation time / s')
    alt_ax.yaxis.set_major_locator(MaxNLocator(3));alt_ax.set_ylim(1.075,1.325)
    yaw_ax.text(.02,.95,'Velocity-aligned yaw + sampled heading offset',transform=yaw_ax.transAxes,
                va='top',fontsize=8.5,color=MUTED,bbox=dict(facecolor='white',edgecolor='none',alpha=.88))

    # C: raw configured ranges, shared normalized axis, actual sampled values.
    param_ax=fig.add_subplot(grid[1,0])
    position=param_ax.get_position()
    label_margin=.085
    param_ax.set_position([position.x0+label_margin,position.y0,position.width-label_margin,position.height])
    title_offset=-label_margin/(position.width-label_margin)
    panel_title(param_ax,'C','Sampled drone parameters')
    param_ax.title.set_position((title_offset,1.0))
    param_ax._left_title.set_position((title_offset,1.0))
    settings=config['randomization'];vehicle=config['vehicle']
    specifications=[
        ('Mass','kg','mass_kg',np.array(settings['mass_scale'])*vehicle['mass_kg'],vehicle['mass_kg']),
        ('Thrust gain','×','thrust_scale',settings['thrust_scale'],1.),
        ('Rate gain','×','rate_scale',settings['rate_scale'],1.),
        ('Rate response','ms','rate_time_constant_s',np.array(settings['rate_time_constant_s'])*1000,vehicle['rate_time_constant_s']*1000),
        ('Thrust response','ms','thrust_time_constant_s',np.array(settings['thrust_time_constant_s'])*1000,vehicle['thrust_time_constant_s']*1000),
        ('Velocity damping','s⁻¹','linear_drag_per_s',settings['linear_drag_per_s'],vehicle['linear_drag_per_s']),
    ]
    for row,(name,unit,key,bounds,reference) in enumerate(specifications):
        low,high=np.asarray(bounds,dtype=float);y=len(specifications)-1-row
        scale=1000 if unit=='ms' else 1.
        normalize=lambda value:(value-low)/(high-low)
        param_ax.plot([0,1],[y,y],color='#e6edf2',lw=9,solid_capstyle='round',zorder=1)
        values=[e['plant'][key]*scale for e in episodes]
        param_ax.scatter([normalize(v) for v in values],[y]*len(values),s=13,color='#a8bac8',alpha=.65,zorder=2)
        for offset,index,color in zip([.13,0.,-.13],selected,COLORS):
            value=episodes[index]['plant'][key]*scale
            param_ax.scatter(normalize(value),y+offset,s=35,color=color,edgecolors='white',linewidths=.7,zorder=4)
        param_ax.scatter(normalize(reference),y,s=41,marker='D',facecolors='white',edgecolors=INK,linewidths=1.2,zorder=5)
        param_ax.text(-.07,y,name,ha='right',va='center',fontsize=9.5,color=INK)
        precision=4 if unit=='kg' else 0 if unit=='ms' else 2
        param_ax.text(1.07,y,f'{low:.{precision}f}–{high:.{precision}f} {unit}',va='center',fontsize=9,color=MUTED)
    param_ax.set_xlim(-.02,1.5);param_ax.set_ylim(-.7,len(specifications)-.3)
    param_ax.set_yticks([]);param_ax.set_xticks([0,.5,1],['Lower bound','Mid-range','Upper bound'])
    param_ax.tick_params(axis='x',colors=MUTED,labelsize=8.5,length=0,pad=10)
    for spine in param_ax.spines.values():spine.set_visible(False)
    param_ax.text(title_offset,-.16,'White diamonds: nominal MPC model.  Colored dots: sampled actual plants.',
                  transform=param_ax.transAxes,fontsize=8.5,color=MUTED)

    # D: explicit data-label timing, not an illustration of trained results.
    flow=fig.add_subplot(grid[1,1]);flow.set_axis_off();flow.set_xlim(0,1);flow.set_ylim(0,1)
    panel_title(flow,'D','From sampled conditions to training pairs')
    items=[(.80,'01','Sample an episode','Task + initial errors + actual dynamics'),
           (.56,'02','Capture the current state','Render RGB at the pre-action drone pose'),
           (.32,'03','Query the MPC expert','Nominal prediction model → thrust + body rates'),
           (.08,'04','Save the synchronized pair','RGB + state + reference → expert action label')]
    for y,number,title,description in items:
        flow.add_patch(FancyBboxPatch((.03,y),.91,.17,boxstyle='round,pad=0.012,rounding_size=0.025',
            facecolor='#f7fafc',edgecolor='#dce5ec',linewidth=1.))
        flow.text(.07,y+.092,number,va='center',fontsize=12,fontweight='bold',color=COLORS[1])
        flow.text(.16,y+.11,title,va='center',fontsize=10.5,fontweight='bold',color=INK)
        flow.text(.16,y+.055,description,va='center',fontsize=8.4,color=MUTED)
        if y>.08:
            flow.add_patch(FancyArrowPatch((.48,y-.015),(.48,y-.055),arrowstyle='-|>',mutation_scale=11,
                                           color='#9eb0bf',linewidth=1.2))
    flow.text(.5,-.10,'Apply the action, advance the dynamics, then repeat steps 02–04.',ha='center',fontsize=8.7,color=MUTED)

    fig.text(.065,.043,'SAMPLED ONCE PER EPISODE',fontsize=9,fontweight='bold',color=COLORS[1])
    fig.text(.065,.022,'MPC receives the reference and state; sampled physical parameters stay hidden. Appearance is unchanged.',
             fontsize=9,color=MUTED)
    fig.text(.96,.043,'Reference/settings visualization — no MPC rollout shown',ha='right',fontsize=9,color=MUTED)
    fig.text(.96,.022,'Source: quadrotor_nmpc.json + randomization.py',ha='right',fontsize=8.5,color=MUTED)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['png','pdf','svg']:
        path=args.output.with_suffix('.'+extension)
        fig.savefig(path,dpi=240,facecolor='white')
        print(path)
    record=dict(seed=args.seed,episodes=episodes,highlighted_seeds=[episodes[i]['seed'] for i in selected],
                config=str(args.config),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                description='Actual randomization samples and reference functions; no MPC or learned-policy rollouts')
    args.output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    plt.close(fig)


if __name__=='__main__':
    main()
