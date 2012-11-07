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
def plot(*arg):
    try:
        matplotlib.rcParams.update({'font.size': 11})

        major = DayLocator()
        minor = HourLocator()
        majorFormat = DateFormatter('%Y-%m-%d')

        fig = figure(figsize=[8,10])
        subplots_adjust(hspace=0.001)

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
                else:
                    if (len(arg[p]) == 2 and i == 1 and
                        (max(abs(arg[p][i-1]['y']))/max(abs(arg[p][i]['y'])) > 3 or
                         max(abs(arg[p][i]['y']))/max(abs(arg[p][i-1]['y'])) > 3)):
                        axt = ax.twinx()
                        axt.plot(t, arg[p][i]['y']*arg[p][i]['factor'], **kwargs)
                        for tl in axt.get_yticklabels():
                            tl.set_color(kwargs['color'])
                        if 'color' in arg[p][i-1]:
                            for tl in ax.get_yticklabels():
                                tl.set_color(arg[p][i-1]['color'])
                    else:
                        ax.plot(t, arg[p][i]['y']*arg[p][i]['factor'], **kwargs)
                i = i+1
            p = p+1
            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(majorFormat)
            ax.xaxis.set_minor_locator(minor)
            ax.yaxis.set_label_coords(-0.08, 0.5)
            if p == 1:
                ax1 = ax
            elif not ('z' in arg[p-1][0]):
                ax.set_xlim(ax1.get_xlim())
            if p < len(arg):
                setp(ax.get_xticklabels(), visible=False)
        show()
    except:
        print "Trigger Exception, traceback info forward to log file."
        print data[len(data)-1]
        traceback.print_exc(file = open("./errlog.txt","a"))
        sys.exit(1)

