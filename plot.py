from pylab import *
from numpy import *
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from matplotlib.patches import Patch
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axisartist import GridHelperCurveLinear
from mpl_toolkits.axisartist import ParasiteAxesAuxTrans
from datetime import datetime
import sys, traceback
from mpl_toolkits.axes_grid import make_axes_locatable
import  matplotlib.axes as maxes
from matplotlib.colors import LogNorm
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from matplotlib import rc

# plot in-situ data
def plot(*arg):
    try:
        matplotlib.rcParams.update({'font.size': 11})
#        rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
#        rc('text', usetex=True)

        major = DayLocator()
        minor = HourLocator()
        majorFormat = DateFormatter('%Y-%m-%d')

        fig = figure(figsize=[8,11])
        fig.subplots_adjust(hspace=0.001)

        p = 0
        ax1 = []
        while p < len(arg):
            ax = fig.add_subplot(len(arg), 1, p+1)
            i = 0
            while i < len(arg[p]):
                t = []
                for k in xrange(len(arg[p][i]['t'])):
                    t.append(datetime.utcfromtimestamp(arg[p][i]['t'][k]))
                kwargs = {}
                if 'marker' in arg[p][i]:
                    kwargs['marker'] = arg[p][i]['marker']
                if len(arg[p]) > 1 and 'color' in arg[p][i]:
                    kwargs['color'] = arg[p][i]['color']
                else:
                    kwargs['color'] = 'black'
                if 'linestyle' in arg[p][i]:
                    kwargs['linestyle'] = arg[p][i]['linestyle']
                if 'z' in arg[p][i]:
                    z = arg[p][i]['z'].reshape(len(arg[p][i]['y']),
                                               len(arg[p][i]['t']))
                    z = z[::-1]
                    t,y = meshgrid(arg[p][i]['t'], arg[p][i]['y'])
                    im = ax.imshow(z*arg[p][i]['factor'], aspect='auto',
                                   cmap='RdBu_r', norm=LogNorm(),
                                   extent=[0, len(arg[p][i]['t'])-1, 0, 180])
                    Bbox = matplotlib.transforms.Bbox.from_bounds(1.02, 0,
                                                                  0.03, 1)
                    trans = ax.transAxes + fig.transFigure.inverted()
                    l, b, w, h = matplotlib.transforms.TransformedBbox(Bbox, trans).bounds
                    cax = fig.add_axes([l, b, w, h])
                    cb = colorbar(im, cax = cax)
                    cb.locator = MaxNLocator(5)
                    cb.update_ticks()
                    axc = ax
                    axc.yaxis.set_label_coords(-0.08, 0.5)
                elif (len(arg[p]) == 2 and i == 1 and
                    (max(abs(arg[p][i-1]['y']))/max(abs(arg[p][i]['y'])) > 3 or
                     max(abs(arg[p][i]['y']))/max(abs(arg[p][i-1]['y'])) > 3)):
                    axt = ax.twinx()
                    setp(ax.get_xticklabels(), visible=False)
                    axt.plot(t, arg[p][i]['y']*arg[p][i]['factor'], **kwargs)
                    if 'color' in arg[p][i]:
                        for tl in axt.get_yticklabels():
                            tl.set_color(arg[p][i]['color'])
                        axt.yaxis.label.set_color(arg[p][i]['color'])
                    if 'color' in arg[p][i-1]:
                        for tl in ax.get_yticklabels():
                            tl.set_color(arg[p][i-1]['color'])
                        ax.yaxis.label.set_color(arg[p][i-1]['color'])
                    axc = axt
                else:
                    axc = ax
                    axc.yaxis.set_label_coords(-0.08, 0.5)
                    if arg[p][i]['name'] == 'beta' or arg[p][i]['name'] == 'Tp':
                        axc.semilogy(t, arg[p][i]['y']*arg[p][i]['factor'], **kwargs)
                    else:
                        axc.plot(t, arg[p][i]['y']*arg[p][i]['factor'], **kwargs)
                if (arg[p][i]['name'] == 'Bx' or arg[p][i]['name'] == 'Br' or
                    arg[p][i]['name'] == 'By' or arg[p][i]['name'] == 'Bt' or
                    arg[p][i]['name'] == 'Bz' or arg[p][i]['name'] == 'Bn' or
                    arg[p][i]['name'] == 'B'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
                    if (len(arg[p]) == 1):
                        axc.set_ylabel('$'+arg[p][i]['name']+r"$ $\mathrm{[nT]}$")
                    else:
                        axc.set_ylabel(r"$B$ $\mathrm{[nT]}$")
                elif (arg[p][i]['name'] == 'Vx' or arg[p][i]['name'] == 'Vr' or
                      arg[p][i]['name'] == 'Vy' or arg[p][i]['name'] == 'Vt' or
                      arg[p][i]['name'] == 'Vz' or arg[p][i]['name'] == 'Vn' or
                      arg[p][i]['name'] == 'Vp'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
                    if (len(arg[p]) == 1):
                        if arg[p][i]['name'] == 'Vp':
                            axc.set_ylabel(r"$V_p$ $\mathrm{[km/s]}$")
                        else:
                            axc.set_ylabel('$'+arg[p][i]['name']+r"$ $\mathrm{[km/s]}$")
                    else:
                        axc.set_ylabel(r"$V_p$ $\mathrm{[km/s]}$")
                elif (arg[p][i]['name'] == 'Pth'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1,2,4], prune='upper'))
                    axc.set_ylabel(r"$P_{th}$ $\mathrm{[nPa]}$")
                elif (arg[p][i]['name'] == 'Pt'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1,2,4], prune='upper'))
                    axc.set_ylabel(r"$P_{t}$ $\mathrm{[nPa]}$")
                elif (arg[p][i]['name'] == 'Pb'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1,2,4], prune='upper'))
                    axc.set_ylabel(r"$P_{b}$ $\mathrm{[nPa]}$")
                elif (arg[p][i]['name'] == 'Np'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                    axc.set_ylabel(r"$N_p$ $\mathrm{[cm^{-3}]}$")
                elif (arg[p][i]['name'] == 'Tp'):
                    axc.yaxis.set_major_locator(FixedLocator(locs=[1e3,1e4,1e5]))
                    axc.set_ylabel(r"$T_p$ $\mathrm{[K]}$")
                elif (arg[p][i]['name'] == 'Vth'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                    axc.set_ylabel(r"$V_{th}$ $\mathrm{[km/s]}$")
                elif (arg[p][i]['name'] == 'beta'):
                    axc.yaxis.set_major_locator(FixedLocator(locs=[1e0,1e1]))
                    axc.set_ylabel('$\\beta$')
                elif (arg[p][i]['name'] == 'AE'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                    axc.set_ylabel(r"$AE$ $\mathrm{[nT]}$")
                elif (arg[p][i]['name'] == 'thetaB'):
                    axc.yaxis.set_major_locator(FixedLocator(locs=[-90,-45,0,45]))
                    ylim([-90,90])
                    axc.set_ylabel('$\\theta_B$')
                elif (arg[p][i]['name'] == 'phiB'):
                    axc.yaxis.set_major_locator(FixedLocator(locs=[0,45,90,135]))
                    ylim([0,180])
                    axc.set_ylabel('$\\phi_B$')
                elif (arg[p][i]['name'] == 'PA'):
                    axc.yaxis.set_major_locator(FixedLocator(locs=[45,90,135]))
                    axc.set_ylabel('$PA$')
                elif (arg[p][i]['name'] == 'He/p'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                    axc.set_ylabel('$He/p$')
                elif (arg[p][i]['name'] == 'O7/O6'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                    axc.set_ylabel('$O7/O6$')
                elif (arg[p][i]['name'] == '<Q>Fe'):
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                    axc.set_ylabel('$<Q>Fe$')
                else:
                    axc.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                    axc.set_ylabel('$'+arg[p][i]['name']+'$')
                i = i+1
            p = p+1
            axc.xaxis.set_major_locator(major)
            axc.xaxis.set_major_formatter(majorFormat)
            axc.xaxis.set_minor_locator(minor)
            if p == 1:
                ax1 = axc
            elif not ('z' in arg[p-1][0]):
                axc.set_xlim(ax1.get_xlim())
            if p < len(arg):
                setp(axc.get_xticklabels(), visible=False)
        show()
    except:
        print "Trigger Exception, traceback info forward to log file."
        traceback.print_exc(file = open("./errlog.txt","a"))
        sys.exit(1)

