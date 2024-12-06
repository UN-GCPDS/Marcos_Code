from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from gcpds.visualizations.topoplots import topoplot


def violin_plot(arrays: ArrayLike,
                positions: ArrayLike = None,
                color:str = 'white',
                ax:Axes = None,
                **kwargs) -> Axes:
    """Make a violin plot from a set of arrays.

    Dots represents each data point while the white dot shows the median value.

    Parameters
    ----------
    arrays : ArrayLike
        Array-like set.
    positions : ArrayLike
        Array of positions. If None, then a linear array is used. By dault None.
    color : str
        Violins' facecolor.
    ax : Axes, optional
        Axes object to which the plot will be added to. If None then one is created. By default None.
    

    Returns
    -------
    Axes
        Matplotlib Axes object

    Notes
    -------
    showextrema keyword is hard-set to False for aesthetic purposes.
    """
    if ax == None:
        fig, ax = plt.subplots()

    if positions == None:
        positions = np.arange(1, len(arrays)+1)
    
    if type(positions) == list:
        positions = np.array(positions)

    if positions.shape[0] != len(arrays):
        raise ValueError(f'Number of Positions ({positions.shape[0]}) does not match the number of arrays ({len(arrays)})')

    if 'showextrema' in kwargs.keys():
        kwargs.pop('showextrema')

    violins = ax.violinplot(arrays, positions = positions, showextrema=False, **kwargs,)
    for position, list_values in zip(positions, arrays):
        min_val = np.min(list_values)
        max_val = np.max(list_values)
        ax.vlines(position, min_val, max_val, color='black')

        median = np.median (list_values)
        ax.scatter([position]*len(list_values), list_values, c='black')
        ax.scatter(position, median, c='white', zorder=3, s=100, edgecolors='k', linewidths=2.0)

    for violin in violins['bodies']:
        violin.set_facecolor(color)
        violin.set_edgecolor('black')
        violin.set_alpha(1)
    return ax

def barplot_stds(locs:ArrayLike,
                 means:ArrayLike,
                 stds:ArrayLike,
                 barwidth:float = 0.8,
                 lineratio:float = 0.4,
                 orientation = 'vertical',
                 barkw:dict = {},
                 linekw:dict = {'colors':'black'},
                 ax=None) -> Axes:
    """Plot a barplot with standard deviation

    Parameters
    ----------
    locs : ArrayLike
        Bar locations
    means : ArrayLike
        The means i.e. the size of the bar.
    stds : ArrayLike
        The standard deviations.
    barwidth : float, optional
        Width for the bars. If orientation is horizontal then it controls the height keyword. 0.8 by default.
    lineratio : float, optional
        Ratio of the line-ends to barwidth. 1.0 means equal size while 0.0 means no line-ends.
    barkw : dict, optional
        Dictionary containing the keyword arguments for the barplot.
    linekw : dict, optional
        Dictionary containing the keyword arguments for the bars.
    orientation : str, optional
        Barplot orientation, vertical by default.
    ax : Axes, optional
        Axes object to which the plot will be added to. If None then one is created, by default None.
        
    Returns
    -------
    Axes
        Matplotlib Axes object
    """
    if ax == None:
        fig, ax = plt.subplots()

    if 'colors' not in linekw.keys():
        linekw['colors'] = 'black'

    lineratio /= 2

    if orientation.lower() in ['vertical', 'v']:
        if 'width' in barkw.keys():
            barkw.pop('width')
        ax.bar(locs, means, barwidth, **barkw)
        ax.vlines(locs, means-stds, means+stds, **linekw)
        ax.hlines(means-stds, locs-barwidth*lineratio, locs+barwidth*lineratio, **linekw)
        ax.hlines(means+stds, locs-barwidth*lineratio, locs+barwidth*lineratio, **linekw)

    elif orientation.lower() in ['horizontal', 'h']:
        if 'height' in barkw.keys():
            barkw.pop('height')
        ax.barh(locs, means, barwidth, **barkw)
        ax.hlines(locs, means-stds, means+stds, **linekw)
        ax.vlines(means-stds, locs-barwidth*lineratio, locs+barwidth*lineratio, **linekw)
        ax.vlines(means+stds, locs-barwidth*lineratio, locs+barwidth*lineratio, **linekw)
    else:
        raise ValueError('Orientation argument only recieves one of the following: \'vertical\', \'v\', \'horizontal\', \'h\'')

    return ax

from gcpds.visualizations.topoplots import topoplot

def multi_topoplot(nrows:int,
                   ncols:int, 
                   *args, 
                   channels: ArrayLike, 
                   montage : str = 'standard_1020', 
                   normalize : bool = True,
                   colorbar : bool = True,
                   **kwargs) -> tuple[Figure, Axes]:
    """Plot multiple topoplots in the same figure.

    Parameters
    ----------
    nrows : int
        Number of rows
    ncols : int
        Number of columns
    channels : ArrayLike
        List of channels. If only one list is given, then all topomaps use the same list.
    montage : str, optional
        Montage of each topomap, if only one is given, then all topomaps use the same. By default 'standard_1020'
    normalize : bool, optional
        Whether to perform Min-Max normalization across topomaps or not.
        This normalization uses the min and max across all topomaps. By default True
    colorbar : bool, optional
        Whether to include a colorbar or not, by default True

    Returns
    -------
    Matplotlib.figure.Figure
        Matplotlib Figure object. This holds information of the figure as a whole.
    Matplotlib.axes.Axes
        Matplotlib Axes objects. This holds information of each topomap.

    Notes
    -------
    Colorbar becomes buggy if there is only one row. Current code only supports up to 4 topomaps in a single row before colorbar size becomes unstable.

    Examples
    -------
    Different channel configurations:
    >>> topo_1 = np.array([1.0, 2.0, 3.0])
    >>> topo_2 = np.array([3.0, 2.0, 1.0])
    >>> ch_1 = ['AFz', 'Cz', 'POz']
    >>> ch_2 = ['T7', 'Cz', 'T8']
    >>> fig, ax = plots.multi_topoplot(1, 2, topo_1, topo_2, channels=(ch_1, ch_2), normalize=True, cmap='viridis')
    >>> plt.show()

    Same channel configuration, but with an empty axes.
    >>> topo_1 = np.array([1.0, 2.0, 3.0])
    >>> topo_2 = np.array([3.0, 2.0, 1.0])
    >>> topo_3 = np.array([3.0, 2.0, 4.0])
    >>> ch = ['AFz', 'Cz', 'POz']
    >>> fig, ax_tp = plots.multi_topoplot(2, 2, topo_1, topo_2,topo_3, channels=ch, normalize=True, cmap='viridis')
    >>> # Remove the 4th, empty axes.
    >>> ax_tp[1, 1].remove()
    >>> # Get the position of the bottom topomap
    >>> x,y,w,h = ax_tp[1, 0].get_position().bounds
    >>> # Move the topomap to the right
    >>> ax_tp[1, 0].set_position([x*2.3,y,w,h])
    >>> plt.show()

    With a topomaps array
    >>> topomaps = np.array([[0.0, 2.0, 5.0],
    >>>                      [0.4, 5.5, 0.5],
    >>>                      [5.2, 1.2, 0.9],
    >>>                      [2.1, 3.4, 1.1]])
    >>> ch = ['AFz', 'Cz', 'POz']
    >>> fig, ax_tp = plots.multi_topoplot(2, 2, *topomaps, channels=ch, normalize=True, cmap='viridis')
    >>> plt.show()
    """
    # Convert montage from string to array-like
    if type(montage) == str:
        montage = np.array([montage])

    # Number of everything
    n_tp = len(args)
    n_ch = len(np.array(channels).shape)
    n_mt = len(montage)
    
    if nrows*ncols < n_tp:
        raise ValueError(f'Not enough axes for plot. Number of topomaps: {n_tp}; Number of Axes: {nrows*ncols}')

    # If only one channel config is found, repeat it for all topomaps
    if n_ch == 1:
        channels = np.array(channels).reshape(1,-1)
        channels = np.repeat(channels, n_tp, axis=0)
        n_ch = n_tp
    elif n_ch != n_tp:
        raise RuntimeError(f'Number of Topomaps ({n_tp}) and Channels configurations ({n_ch}) does not match')
    
    # If only one montage config is found, repeat it for all topomaps
    if n_mt == 1:
        montage = np.array(montage).reshape(1,-1)
        montage = np.repeat(montage, n_tp, axis=0).squeeze()
        n_mt = n_tp
    elif n_mt != n_tp:
        raise RuntimeError(f'Number of Topomaps ({n_tp}) and Montage configurations ({n_mt}) does not match')
    
    # If number of channel configurations and montages does not coincide, raise an error.
    if n_mt != n_ch:
        raise RuntimeError(f'Number of Channel configurations ({n_ch}) and Montage configurations ({n_mt}) does not match')

    # Find the vmin and vmax across all topomaps
    min_val = np.min(args)
    max_val = np.max(args) - min_val*normalize # If normalize is false, then this is equivalent to performing max_val - 0

    # Set vlim limits.
    if normalize:
        limits = (0, 1)
    else:
        limits = (min_val, max_val)

    # If we only get one topomap, we need to turn axes into an array.
    fig, ax = plt.subplots(nrows,ncols)
    if nrows==ncols==1:
        ax = np.array([ax])

    # We advance as long as we have topomaps to plot.
    # We fill subplots from left to right, top to bottom.
    i, j = 0, 0
    for channel, mt, array in zip(channels, montage, args):
        if normalize:
            array_ = array-min_val
            array_ /= max_val
        else:
            array_ = array
        if nrows == 1:
            im,_ = topoplot(array_, channels=list(channel), ax=np.array(ax)[j], vlim=limits, show=False, montage_name=mt, **kwargs)
        elif ncols == 1:
            im,_ = topoplot(array_, channels=list(channel), ax=np.array(ax)[i], vlim=limits, show=False, montage_name=mt, **kwargs)
        else:
            im,_ = topoplot(array_, channels=list(channel), ax=np.array(ax)[i,j], vlim=limits, show=False, montage_name=mt, **kwargs)

        j += 1
        # If we've reached the last column, restart from the next row
        if j == ncols:
            j = 0
            i += 1

    if colorbar:
        if nrows==1:
            # We add a new axes to prevent the last topomap becoming smaller
            cbax = fig.add_axes([0.93, #x-position
                                 -0.025*ncols**2+0.213*ncols-0.085, # y-position
                                 0.03, # width
                                 0.0525*ncols**2-0.4275*ncols+1.1225])#height
            
            # "Make the inside of the axes the colorbar"
            fig.colorbar(im, cax=cbax)
        else:
            # If we have more than one row we don't have to worry about topomaps becoming smaller
            fig.colorbar(im, ax=ax.ravel().tolist()) 

    return fig, ax
        