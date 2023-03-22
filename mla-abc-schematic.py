'''
Create schematic diagram for MLA-ABC paper
'''

import sciris as sc
import numpy as np
import pylab as pl
import scipy.stats as st

#%% Define distance

def distance(xpts, ypts):
    
    bestx = np.array([0.45 , 0.465, 0.525, 0.6  , 0.705])
    diffs = sc.cat(0, np.cumsum(np.diff(bestx)[::-1]))
    besty = 0.5 + diffs
    n = len(xpts)
    
    alldists = np.zeros((n, len(bestx)))
    
    for i,(bx,by) in enumerate(zip(bestx, besty)):
        thisdists = ((xpts-bx)**2 + (ypts-by)**2)**1/2
        alldists[:,i] = thisdists
    
    rweight = 0.8
    zpts = 100*alldists.min(axis=1)
    zpts *= np.random.uniform(1-rweight, 1+rweight, n)
    zpts = 1/(zpts+1)
    
    return zpts


#%% Generate data
n = 500
n_samp = 50
ml_thresh = 0.4
ac_thresh = 0.8
xlim = [0,1]
ylim = xlim
np.random.seed(1)
lhs = st.qmc.LatinHypercube(d=2)
sample = lhs.random(n=n)
xpts = sample[:,0] # np.random.uniform(xlim[0], xlim[1], size=n)
ypts = sample[:,1] # np.random.uniform(ylim[0], ylim[1], size=n)
zpts = distance(xpts, ypts)
cmap = 'viridis'
colors = sc.vectocolor(zpts, cmap=cmap)

tr_samp = lhs.random(n=n_samp) # tr_inds = np.random.choice(n, n_samp)
xsamp = tr_samp[:,0]
ysamp = tr_samp[:,1]
zsamp = distance(xsamp, ysamp)
csamp = sc.vectocolor(zsamp, cmap=cmap)

ml_inds = zpts > ml_thresh
ac_inds = zpts > ac_thresh



#%% Plotting
sc.options(dpi=200, font='Libertinus Sans', fontsize=12)

fig = pl.figure(figsize=(10,10))

mainxl = 0.07
mainxr = 0.57
mainy1 = 0.7
mainy2 = 0.37
mainy3 = 0.05
wmain = 0.37
hmain = 0.23
dsmall = 0.03
δ = 0.02
kw = dict(pad=30, fontsize=15)

def small_ax(xpos, ypos, x, y, buff=0.005):
    axt = pl.axes([xpos, ypos+hmain+buff, wmain, dsmall])
    axr = pl.axes([xpos+wmain+buff, ypos, dsmall, hmain])
    
    fbkw = dict(alpha=0.5)
    pts = np.linspace(*xlim)
    for orient,data,ax in zip(['x','y'], [x,y], [axt,axr]):
        kde = st.gaussian_kde(sorted(data))
        pdf = kde.pdf(pts)
        if orient == 'x':
            ax.fill_between(pts, pdf, 0, **fbkw)
            ax.set_xlim(xlim)
        else:
            ax.fill_betweenx(pts, pdf, 0, **fbkw)
            ax.set_ylim(ylim)
        ax.axis('off')
    
    return ax


def plot_mods(ax=None, labs=False):
    if ax is not None: pl.sca(ax)
    pl.xlim([-δ, 1+δ])
    pl.ylim([-δ, 1+δ])
    if labs:
        pl.xlabel('Parameter 1')
        pl.ylabel('Parameter 2')
    else:
        pl.xticks([], [])
        pl.yticks([], [])
        
    return


def scatter(x, y, c, inds=...):
    pl.scatter(x[inds], y[inds], c=c[inds], alpha=0.7)
    return

# R-ABC initial
ax1 = pl.axes([mainxl, mainy2, wmain, hmain])
scatter(xpts, ypts, colors)
plot_mods()
small_ax(mainxl, mainy2, x=xpts, y=ypts)
ax1.set_title('R-ABC: 1. Samples (uninformed prior)', **kw)

# R-ABC final
ax2 = pl.axes([mainxl, mainy3, wmain, hmain])
scatter(xpts, ypts, colors, ac_inds)
plot_mods(labs=True)
small_ax(mainxl, mainy3, x=xpts[ac_inds], y=ypts[ac_inds])
ax2.set_title('R-ABC: 2. Keep good samples (posterior)', **kw)


# Color bar
cax = pl.axes([mainxl+wmain*0.2, mainy1+hmain*0.9, wmain*0.6, hmain*0.2])
pl.colorbar(ax=ax1, cax=cax, orientation='horizontal')
cax.set_xticks([0, 1], ['Low', 'High'])
cax.set_title('Likelihood')

# Data plots
nd = 25
time = np.linspace(0, 1, nd)
data = np.exp(-(((time-0.3)*5)**2)) + 3*np.exp(-(((time-0.7)*6)**2)) + 0.5*np.random.rand(nd)
bad_fit  = 2*np.exp(-(((time-0.4)*4)**2)) + 2*np.exp(-(((time-0.6)*10)**2))
good_fit = 1.1*np.exp(-(((time-0.3)*5)**2)) + 3.2*np.exp(-(((time-0.7)*6)**2))
simax1 = pl.axes([mainxl-wmain*0.05, mainy1+hmain*0.1, wmain*0.4, hmain*0.5])
simax2 = pl.axes([mainxl+wmain*0.65, mainy1+hmain*0.1, wmain*0.4, hmain*0.5])
for ax in [simax1, simax2]:
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.scatter(time, data, c='k', alpha=0.5, label='Data')
    ax.set_xlim([0,1])
pkw = dict(lw=3, alpha=0.8)
simax1.plot(time, bad_fit,  c=colors[np.argmin(zpts)], label='Model', **pkw)
simax2.plot(time, good_fit, c=colors[np.argmax(zpts)], label='Model', **pkw)
simax1.set_xlabel('Time')
simax1.set_ylabel('Infections')
simax1.legend(frameon=False, bbox_to_anchor=(1.0, 0.8))
simax1.set_title('Low likelihood')
simax2.set_title('High likelihood')


# ML-ABC training
ax3 = pl.axes([mainxr, mainy1, wmain, hmain])
scatter(xsamp, ysamp, csamp)
plot_mods()
small_ax(mainxr, mainy1, x=xsamp, y=ysamp)
ax3.set_title('ML-ABC: 1. Training data (uninformed prior)', **kw)

# ML-ABC initial
ax4 = pl.axes([mainxr, mainy2, wmain, hmain])
scatter(xpts, ypts, colors, ml_inds)
plot_mods()
small_ax(mainxr, mainy2, x=xpts[ml_inds], y=ypts[ml_inds])
ax4.set_title('ML-ABC: 2. Choose likely samples (informed prior)', **kw)

# ML-ABC final
ax5 = pl.axes([mainxr, mainy3, wmain, hmain])
scatter(xpts, ypts, colors, ac_inds)
plot_mods(labs=True)
small_ax(mainxr, mainy3, x=xpts[ac_inds], y=ypts[ac_inds])
ax5.set_title('ML-ABC: 3. Keep good samples (posterior)', **kw)


# Finish up
sc.savefig('mla-abc-schematic.png')
pl.show()
print('Done')