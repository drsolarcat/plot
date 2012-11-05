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



# plot in-situ data
def plot(year, month, day, hour, minute, second, panels, *arg):
    try:
        matplotlib.rcParams.update({'font.size': 11})

        dates = []

        for i in xrange(len(year)):
            dates.append(datetime(year[i], month[i], day[i],
                                  hour[i], minute[i], second[i]))

        major = DayLocator()
        minor = HourLocator()
        majorFormat = DateFormatter('%Y-%m-%d')

        fig = figure(figsize=[8,10])
        subplots_adjust(hspace=0.001)

        p = 0
        i = 0
        while i < len(arg):
            p = p+1
            ax = subplot(panels, 1, p)
            for k in xrange(arg[i]):
                name = arg[i+(k+1)*2-1]
                data = arg[i+(k+1)*2]
                if (name == 'Bx' or name == 'Br' or name == 'Vx' or name == 'Vr' or
                    name == 'sVx' or name == 'sVr'):
                    color = 'r'
                elif (name == 'By' or name == 'Bt' or name == 'Vy' or
                      name == 'Vt' or name == 'sVy' or name == 'sVt'):
                    color = 'g'
                elif (name == 'Bz' or name == 'Bn' or name == 'Vz' or
                      name == 'Vn' or name == 'sVz' or name == 'sVn'):
                    color = 'b'
                else:
                    color = 'k'
                factor = 1
                if (name == 'B' or name == 'Bx' or name == 'Br' or name == 'By' or
                    name == 'Bt' or name == 'Bz' or name == 'Bn'):
                    factor = 1e9
                if (name == 'Vp' or name == 'Vx' or name == 'Vr' or name == 'Vy' or
                    name == 'Vt' or name == 'Vz' or name == 'Vn' or name == 'Vth' or
                    name == 'sVp' or name == 'sVx' or name == 'sVr' or
                    name == 'sVy' or name == 'sVt' or name == 'sVz' or
                    name == 'sVn'):
                    factor = 1e-3
                if (name == 'Pth' or name == 'Pt'):
                    factor = 1e9
                if (name == 'Np'):
                    factor = 1e-6
                if (name == 'Tp' or name == 'beta'):
                    ax.semilogy(dates, data*factor, color)
                elif (name == 'PA'):
                    angles = array(data[:20])
                    pads = array(data[20:])
    #                pads[pads <= 0] = min(pads[pads > 0])
                    pa = pads.reshape(20, len(year))
                    pa = pa[::-1]
                    x,y = meshgrid(arange(0, len(year)), angles)
                    try:
                        im = ax.imshow(pa, aspect='auto', cmap='RdBu_r', norm=LogNorm(), extent=[0, len(year)-1, 0, 180])
                        Bbox = matplotlib.transforms.Bbox.from_bounds(1.02, 0, 0.03, 1)
                        trans = ax.transAxes + fig.transFigure.inverted()
                        l, b, w, h = matplotlib.transforms.TransformedBbox(Bbox, trans).bounds
                        cax = fig.add_axes([l, b, w, h])
                        cb = colorbar(im, cax = cax)
                        cb.locator = MaxNLocator(5)
                        cb.update_ticks()
                    except:
                        print "Trigger Exception, traceback info forward to log file."
                        traceback.print_exc(file = open("./errlog.txt","a"))
                        sys.exit(1)
                elif (name == 'He/p'):
                    t = []
                    i = i+1
                    for m in xrange(len(data)):
                        t.append(datetime.utcfromtimestamp(data[m]))
                    d = arg[i+(k+1)*2]
                    ax.plot(t, d, color)
                else:
                    ax.plot(dates, data*factor, color)
                if (name == 'Bx' or name == 'Br' or name == 'By' or name == 'Bt' or
                    name == 'Bz' or name == 'Bn' or name == 'B'):
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
    #                ax.yaxis.set_minor_locator(MultipleLocator(1))
                    if (arg[i] == 1):
                        ax.set_ylabel('$'+name+'$ $[nT]$')
                    else:
                        ax.set_ylabel('$B$ $[nT]$')
                if (name == 'Vx' or name == 'Vr' or name == 'Vy' or name == 'Vt' or
                    name == 'Vz' or name == 'Vn' or name == 'Vp' or name == 'sVp' or
                    name == 'sVx' or name == 'sVr' or name == 'sVy' or
                    name == 'sVt' or name == 'sVz' or name == 'sVn'):
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
                    if (arg[i] == 1):
                        ax.set_ylabel('$'+name+'$ $[km/s]$')
                    else:
                        ax.set_ylabel('$V_p$ $[km/s]$')
                if (name == 'Pth'):
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1,2,4], prune='upper'))
                    ax.set_ylabel('$P_{th}$ $[nPa]$')
                if (name == 'Pt'):
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1,2,4], prune='upper'))
                    ax.set_ylabel('$P_t$ $[nPa]$')
                if (name == 'Np'):
                    ax.xaxis.set_minor_locator(minor)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                    ax.set_ylabel('$N_p$ $[cm^{-3}]$')
                if (name == 'Tp'):
                    ax.yaxis.set_major_locator(FixedLocator(locs=[1e3,1e4,1e5]))
                    ax.set_ylabel('$T_p$ $[K]$')
                if (name == 'Vth'):
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                    ax.yaxis.set_minor_locator(MultipleLocator(5))
                    ax.set_ylabel('$V_{th}$ $[km/s]$')
                if (name == 'beta'):
                    ax.yaxis.set_major_locator(FixedLocator(locs=[1e0,1e1]))
                    ax.set_ylabel('$\\beta$')
                if (name == 'AE'):
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                    ax.set_ylabel('$AE$ $[nT]$')
                if (name == 'thetaB'):
                    ax.yaxis.set_major_locator(FixedLocator(locs=[-90,-45,0,45]))
                    ylim([-90,90])
                    ax.set_ylabel('$\\theta_B$')
                if (name == 'phiB'):
                    ax.yaxis.set_major_locator(FixedLocator(locs=[0,45,90,135]))
                    ylim([0,180])
                    ax.set_ylabel('$\phi_B$')
                if (name == 'PA'):
                    ax.yaxis.set_major_locator(FixedLocator(locs=[45,90,135]))
                    ax.set_ylabel('$PA$')
            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(majorFormat)
            ax.xaxis.set_minor_locator(minor)
            ax.yaxis.set_label_coords(-0.08, 0.5)
            if (p < panels):
                setp(ax.get_xticklabels(), visible=False)
            i = i+1+arg[i]*2
        show()
    except:
        print "Trigger Exception, traceback info forward to log file."
        print data[len(data)-1]
        traceback.print_exc(file = open("./errlog.txt","a"))
        sys.exit(1)

