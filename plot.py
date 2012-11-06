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
        while p < len(arg):
            ax = subplot(len(arg), 1, p+1)
            i = 0
            while i < len(arg[p]):
                t = []
                for k in xrange(len(arg[p][i]['t'])):
                    t.append(datetime.utcfromtimestamp(arg[p][i]['t'][k]))
                ax.plot(t, arg[p][i]['y']*arg[p][i]['factor'])
                i = i+1
            p = p+1
            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(majorFormat)
            ax.xaxis.set_minor_locator(minor)
            ax.yaxis.set_label_coords(-0.08, 0.5)
        show()
    except:
        print "Trigger Exception, traceback info forward to log file."
        print data[len(data)-1]
        traceback.print_exc(file = open("./errlog.txt","a"))
        sys.exit(1)

