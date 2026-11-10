#!/usr/bin/env priithon
import itertools
import numpy as np
N = np
from scipy import interpolate, ndimage as nd

from . import ppro26
from .all import F


## ------ basic processes
        
def regionalProcess(func, arr, npxls=(20,40,40), shift=0, overlap=0.1, full=True, *args, **kwds):
    """
    Parameters
    -----------
    func : callable
        expected to receive subimage as the first argument, and return the result
    arr : ND array
        multidimensional image array
    npxls : iterable or int or float
        number of pixels for a single region, or fraction of array.shape if float
    shift : iterable or int or float
        number of pixels for shifting the start, or fraction of array.shape if float
    overlap : iterable or int or float
        number of pixels for region overlap, or fraction of array.shape if float
    full: if True, then full region is used (the ending regions may have a larger shape)
    args, kwds : list or dict
        arguments or kwds for func

    Retruns:
    -----------
    (slcs, result)
    """

    
    slcs = chopShapeND(arr.shape, npxls, shift, overlap, full)

    #if ppro26.NCPU > 1:
    #    ncpu = F.fftw.ncpu
    #    F.fftw.ncpu = 1
    ret = ppro26.pmap(_func, slcs, func=func, arr=arr, *args, **kwds)
    #if ppro26.NCPU > 1:
    #    F.fftw.ncpu = ncpu
    return slcs, ret
        
def _func(slc, func, arr, *args, **kwds):
    """
    this helper function needs to be outside of the parent function to pickle by multiprocessing
    """
    return func(arr[slc], *args, **kwds)

def chopShapeND(shape, npxls=(32,32), shift=(0,0), overlap=0.1, full=True):
    """
    shape: original image shape
    npxls: number of pixels to divide ((z,)y,x) or scaler
    shiftOrigin: shift the origin in pixels
    full: if True, then full region is used (the ending regions may have a larger shape)
          if False, then all regions have the same shape but edge regions may not be used
    
    return list of [[[start, stop],...,], [[start, stop],...,]]
    i.e. [zslices, yslices, xslices]
    """
    #print('chop', shape, npxls, shift, overlap, full)
    npxls = _checkShape(shape, npxls)
    shift = _checkShape(shape, shift)
    overlap = _checkShape(npxls, overlap)
        
    mimas = []
    for d, s in enumerate(shape):
        ndiv = (s - shift[d]) // (npxls[d] - overlap[d])
        mms = []
        if full:
            start = 0
        else:
            start = (s - ((npxls[d] - overlap[d]) * ndiv)) // 2

        start += shift[d]
        for n in range(int(ndiv)):
            if (start + npxls[d]) > s:
                break
            mms += [[start, start + npxls[d]]]
            start += npxls[d] - overlap[d]
            
        if full:
            mms[-1][-1] = None
            
            
        mimas.append([slice(*m) for m in mms])

    return list(itertools.product(*mimas))


def _checkShape(shape, what):
    """
    This helper function prepares argments to suit with the shape
    if what is float, it is a fraction of the shape
    """
    if hasattr(what, '__iter__'):
        if len(shape) > len(what):
            if N.any([type(w) == float for w in what]):
                what = tuple(shape[:-len(what)]) + tuple([int(round(s)) for s in N.multiply(shape[-len(what):], what)])
            else:
                what = tuple(shape[:-len(what)]) + tuple(what)
        elif len(shape) != len(what):
            raise ValueError('length of the input tuple must be the same as len(shape)')
        elif N.any([type(w) == float for w in what]):
            what = [int(round(s)) for s in N.multiply(what, shape)]
    elif type(what) == int:
        what = [what for d in range(len(shape))]
    elif type(what) == float:
        if what < 0:
            raise ValueError('Negative fraction is not allowed')
        what = [int(round(s * what)) for s in shape]
    return what

def viewRegions3D(shape, slcs, arr=None):
    """
    view in the viewer
    works for shape with up to 3D
    """
    from .all import Y, U

    ndim = len(shape)

    if ndim >= 2:
        if arr is None:
            y, x = N.indices(shape[-2:], dtype=N.float32)
            arr = y + x
        elif arr.ndim > 2:
            a = arr
            while a.ndim > 2:
                a = U.project(a, 0)
        else:
            a = arr
        Y.view(a)
        vy = Y.viewers[-1]
    else:
        if arr is None:
            x = N.arange(shape[-1])
            x /= x.max()
            x *= (2*N.pi) * (shape[-1]//10)
            arr = N.sin(x)
        elif arr.ndim > 1:
            a = arr
            while a.ndim > 1:
                a = U.project(a, 0)
        else:
            a = arr
        Y.ploty(a)
        
    if ndim >= 3:
        if arr is None:
            z, y = N.indices(shape[-3:-1], dtype=N.float32)
            a = y+z
        elif arr.ndim > 2:
            a = arr
            while a.ndim > 2:
                a = U.project(a, 1)
        else:
            a = arr
                
        Y.view(a)
        vz = Y.viewers[-1]

    nslcs = len(slcs) - 1
    for si, slc in enumerate(slcs):
        ss = [slc[i].indices(s)[:2] for i, s in enumerate(shape)]
        
        if ndim >= 2:
            ps = list(zip(*ss[-2:]))
            Y.vgAddRect(vy.id, ps, refreshNow=(si == nslcs))
            
        if ndim >= 3:
            ps = list(zip(*ss[-3:-1]))
            Y.vgAddRect(vz.id, ps, refreshNow=(si == nslcs))



### ---   regional drift functions -------------
def synthesize_regional_arr(slcs, small_arrs, original_shape):
    """
    map small_arrs into the array with the original_shape

    Parameters
    ------------
    slcs : list of slices
        the first product from regionalProcess()
    small_arrs: list of ND arrays
        same length as slcs, it may be the second product from regionalProcess()
    original_shape : iterable
        shape of the original image array

    Returns:
    --------
    An array with the original shape
    """
    nregions = len(slcs)
    canvas = N.empty(original_shape, dtype=N.float32)
    for i, slc in enumerate(slcs):
        if small_arrs[0].ndim == 1:
            cc = canvas[tuple(slc)].T
            cc[:] = small_arrs[i]
            canvas[tuple(slc)] = cc.T
        else:
            try:
                canvas[tuple(slc)] = small_arrs[i]
            except ValueError:
                slc = [slice(s.start, s.start+small_arrs[i].shape[d]) for d, s in enumerate(slc)]
                canvas[tuple(slc)] = small_arrs[i]
    return canvas

def centers_from_slices(shape, slcs):
    """
    This function is separated from the zoom2D function because the shape may not be the same between the two functions.

    Parameters
    ----------
    shape : iterable
        shape of the image array
    slcs : list of slices
        the first product from regionalProcess()

    Returns
    --------
    (a list of the center point coordinates from the slice lists)
    """
    points = []
    shape = shape[-len(slcs[0]):]
    
    for i, slc in enumerate(slcs):
        pt = [N.mean(s.indices(shape[d])[:2]) for d, s in enumerate(slc)]
        points.append(pt)
    return points

def zoom2DMap(yxs, points_yx, shape, start=None):
    """
    Zoom up regional value map to the shape of the array

    Parameters
    ----------
    yxs : array
        shape = (nregion, nz, 2)
    points_yx : array
        output from centers_from_slices
    shape : iterable
        shape of the image array
    start : iterable
        start indices of the region

    Returns
    --------
    Zoomed_up_yxs
    """

    
    if start is not None:
        if hasattr(start, "__iter__") and type(start[-1]) == slice:
            start = N.array([x.start for x in start if x != Ellipsis])
        points_yx = N.array([yx[-2:] + start for yx in points_yx])
    elif len(points_yx[0]) > 2:
        points_yx = N.array([yx[-2:] for yx in points_yx])
    
    shape = shape[-2:]
    # process along the Z axis
    ret = ppro26.pmap(_zoom2DMap, yxs, points_yx=points_yx, shape=shape)
    try:
        ret = N.array(ret)
    except:
        pass
    return ret

def _zoom2DMap(vals_yx, points_yx, shape):
    # interpolate from the point data
    vals = N.array(vals_yx).T # shape (n,)
    points = points_yx # shape (n, D)
    pmin = points[0]
    psize = N.subtract(points[-1], pmin).astype(N.int)
    gyx = N.array(N.indices(psize)).T + pmin
    zoomed = [interpolate.griddata(points, v, gyx, method='cubic').T for v in vals]
    # HACK: transposition after griddata produced the best result

    # treat outside regions
    pad_width = [(m, shape[d]-(m+psize[d])) for d, m in enumerate(pmin.astype(N.int))]
    zoomed = [N.pad(z, pad_width, 'edge') for z in zoomed]
    
    return N.array(zoomed)

def applyRegionalMap(arr, mapyxs, base_yxs=None):
    """
    Correct regional drifts 2D

    Parameters
    -----------
    arr : ND array
    mapyxs : list of arrays
        list of translation maps (nz, nregions, 2, ny, nx)
    base_yxs : arrays
        translation array (nz,ny,nx)

    Returns
    ----------
    corrected array
    """
    if base_yxs is None:
        base_yxs = [None] * len(arr)
    #else:
    #    base_yxs = N.cumsum(base_yxs, axis=0)
    arr_mapyxs = zip(arr, mapyxs, base_yxs)
    ret = ppro26.pmap(_applyRegionalMap, arr_mapyxs)
    try:
        ret = N.array(ret, dtype=N.float32)
    except:
        pass
    return ret

def _applyRegionalMap(arr_mapyx):
    arr, mapyx, base_yxs = arr_mapyx
    ind = N.indices(mapyx[0].shape)
    mapyx = ind - mapyx # the shift direction is opposite in map_coordinates
    if base_yxs is not None:
        mapyx = (mapyx.T - base_yxs).T
    return nd.map_coordinates(arr, mapyx)



class RegionalIterator(object):
    def __init__(self, arr, min_npxls=64, shift=0, overlap=0, full=True):
        """
        Parameters
        -----------
        arr : ND array
            multidimensional image array
        min_nnpxls : iterable or int or float
            number of pixels for the minimum region, or fraction of array.shape if float
        shift : iterable or int or float
            number of pixels for shifting the start, or fraction of array.shape if float
        overlap : iterable or int or float
            number of pixels for region overlap, or fraction of array.shape if float
        full: if True, then full region is used (the ending regions may have a larger shape)

        >>> ri = RegionalIterator(arr, min_npxls, shift, overlap, full)
        >>> ri.set_target_func(your_func)
        >>> ri.set_base_correction_func(your_correction_func)
        >>> ...

        >>> ri.singleRegionalProcess(base_yxs=base_yxs, npxls=(min_npxls,min_npxls), phaseContrast=phaseContrast, threshold=threshold, nsecs=nsecs, parallel=False)
        >>> ri.iterativeRegionalProcess(base_yxs, phaseContrast=phaseContrast, threshold=threshold, nsecs=nsecs, parallel=False)
        """
        self.arr = arr
        self.min_npxls = min_npxls
        self.shift = shift
        self.overlap = overlap
        self.full = full
        self.last_npxls = min_npxls

        self.set_max_niter()
        self.set_target_func()
        self.set_regional_map_func()
        self.set_base_correction_func()
        self.set_crop_func()
        self.set_screen_func()
        self.set_post_process_func()
        self.set_cumsum_func()
        #self.set_debug_func()
        self.set_echo_func()
        self.set_debug()

    def set_max_niter(self, n=10):
        self.max_niter = n
        
    def set_target_func(self, func=None):
        self.target_func = func

    def set_regional_map_func(self, func=applyRegionalMap):
        self.regional_map_func = func

    def set_base_correction_func(self, func=None):
        self.base_correction_func = func

    def set_crop_func(self, func=None):
        """
        crop_func(base_yx, regional_yx=None)
        """
        self.crop_func = func

    def set_screen_func(self, func=None):
        self.screen_func = func

    def set_post_process_func(self, func=None):
        """
        often the func can be self.extract_first_entry()
        """
        self.post_process_func = func

    def set_cumsum_func(self, func=None):
        """
        func can be N.cumsum
        """
        self.cumsum_func = func

    #def set_debug_func(self, func=None):
    #    self.debug_func = func

    def set_echo_func(self, func=print):
        self.echo_func = func

    def set_progress_func(self, func=None):
        self.progressfunc = func
        
    def progress(self):
        if self.progressfunc:
            try:
                next(self.progressfunc)
            except StopIteration:
                self.echo_func('Progress StopIteration')

    def set_debug(self, debug=False, outfnbase=''):
        self.debug = debug
        self.outfnbase = outfnbase
        
    def extract_first_entry(self, ret):
        return [entry[0] for entry in ret]
        

    def screen_regional(self, ret, *args, **kwds):
        ret = ppro26.pmap(self.screen_func, ret, *args, **kwds)
        return ret

    def transpose(self, yxs):
        """
        Parameters
        -----------
        yxs : array
            yxs array with a shape of (region,z,yx)

        Returns 
        -----------
        yxs array with a shape of (z,region,yx)
        """
        axes = list(range(yxs.ndim))
        axes[:2] = axes[:2][::-1]
        yxs = N.transpose(yxs, axes=axes)#(1,0,2)) # (region,z,yx) -> (z,region,yx)
        return yxs
        
    def singleRegionalProcess(self, zoomed_yxs=0, base_yxs=0, npxls=(128,128), *args, **kwds):
        if not self.target_func:
            raise ValueError('Please set target function')

        if not hasattr(npxls, '__iter__'):
            npxls = (int(npxls), int(npxls))

        self.echo_func('correcting before going into current window ', min(npxls))
        if hasattr(zoomed_yxs, '__iter__'):
            corrected = self.regional_map_func(self.arr, zoomed_yxs, base_yxs=base_yxs)
        elif self.base_correction_func:
            corrected = self.base_correction_func(self.arr, base_yxs)
        else:
            corrected = self.arr
        self.echo_func('done initial correcting  ')
            
        if self.crop_func:
            slcs0 = self.crop_func(base_yxs, zoomed_yxs)
            corrected = corrected[slcs0]
        else:
            slcs0 = None

        if self.debug:
            if self.outfnbase:
                from .imgio import Mrc
                out = self.outfnbase + '_before%i.mrc' % min(npxls)
                Mrc.save(corrected, out, ifExists='overwrite')
                self.echo_func('saving before done %i ' % min(npxls))

        self.echo_func('start the regional process, current window=%i ' % min(npxls))

        slcs, yxs = regionalProcess(self.target_func, corrected, npxls, self.shift, self.overlap, self.full, *args, **kwds)
        [self.progress() for i in range(self.arr.shape[-1]//npxls[-1])]
            #for yx in yxs:
            #    for y in yx:
                    #self.progress()
            #[self.progress() for i in range(len(slcs))]
        self.echo_func('done the regional process ')
        
        if self.screen_func:
            yxs = self.screen_regional(yxs, *args, **kwds)

        if self.debug:
            if self.outfnbase:
                from .imgio import Mrc
                import pickle
                #out = self.outfnbase + '_before%i.mrc' % min(npxls)
                #Mrc.save(corrected, out, ifExists='overwrite')
                ys = [x[0][:,0] for x in yxs]
                xs = [x[0][:,1] for x in yxs]
                #print([x[0] for x in yxs])
                xcfs = [x[1] for x in yxs]
                ys = synthesize_regional_arr(slcs, ys, corrected.shape)
                xs = synthesize_regional_arr(slcs, xs, corrected.shape)
                xcfs = synthesize_regional_arr(slcs, xcfs, corrected.shape)
                out = self.outfnbase + '_xcf%i.mrc' % min(npxls)
                #synt = N.array((xcfs, ys, xs))
                #Mrc.save(synt, out, ifExists='overwrite')
                Mrc.save(xcfs, out, ifExists='overwrite')
                print('saving done', min(npxls))

                #yxs0 = [x[0] for x in yxs]
                #with open(self.outfnbase+'_yxs.pkl', 'wb') as o:
                #    pickle.dump(yxs0, o)
                #with open(self.outfnbase+'_baseyx.pkl', 'wb') as o:
                #    pickle.dump(base_yxs, o)
                with open(self.outfnbase+'_slc%i.pkl' % min(npxls), 'wb') as o:
                    pickle.dump((slcs, corrected.shape), o)
            
        if self.post_process_func:
            yxs = self.post_process_func(yxs)

        points = centers_from_slices(corrected.shape, slcs)

        yxs = N.array(yxs)
        yxs = self.transpose(yxs)
        
        if self.cumsum_func:
            yxs = self.cumsum_func(yxs, axis=0)

        if self.debug:
            if self.outfnbase:
                out = self.outfnbase + '_yxs%i.pkl' % min(npxls)
                with open(out, 'wb') as o:
                    pickle.dump(yxs, o)
            
        self.echo_func('Preparing transformation map win=%i ' % min(npxls))
        zoomed_yxs = zoom2DMap(yxs, points_yx=points, shape=self.arr.shape, start=slcs0)
        self.progress()
        self.progress()
        
        self.echo_func('done making map win=%i ' % min(npxls))
        

        self.corrected = corrected
        self.zoomed_yxs = zoomed_yxs
        #return corrected, zoomed_yxs

    def iterativeRegionalProcess(self, base_yxs, *args, **kwds):

        npxls = min(N.divide(self.arr.shape[1:], 4).astype(N.int))
        zyxs = 0

        if npxls < self.min_npxls:
            if self.base_correction_func:
                self.corrected = self.base_correction_func(self.arr, base_yxs)
            else:
                self.corrected = self.arr
            self.last_npxls = min(self.arr.shape[-2:])
            return

        for i in range(self.max_niter):

            self.singleRegionalProcess(zyxs, base_yxs, npxls, *args, **kwds)
            zyxs = zyxs + self.zoomed_yxs
            if hasattr(zyxs, "__iter__"):
                zmax = zyxs.max()
            else:
                zmax = 0
            self.echo_func('done window size=%i shiftmax=%.2f ' % (npxls, zmax))


                    
            
            self.last_npxls = npxls
            npxls //= 2
            if npxls < self.min_npxls:
                break
            
        self.echo_func('applying the final map ')
        self.progress()
        corrected = self.regional_map_func(self.arr, zyxs, base_yxs=base_yxs)

        if self.crop_func:
            slcs0 = self.crop_func(base_yxs, zyxs)
            corrected = corrected[slcs0]

        self.corrected = corrected
        
        self.zyxs = zyxs
        self.base_yxs = base_yxs
        #return corrected

        if self.debug:
            if self.outfnbase:
                from .imgio import Mrc
                out = self.outfnbase + '_map.mrc'
                
                Mrc.save(zyxs.astype(N.float32), out, ifExists='overwrite')
                self.echo_func('saving map done ')

def load(fn):
    import pickle
    with open(fn, 'rb') as h:
        return pickle.load(h)

def findyxpos(yx, slcs, original_shape=(1024,1024)):
    found = False
    for i, slc in enumerate(slcs):
        if all([(s.indices(original_shape[d])[0] <= yx[d] and s.indices(original_shape[d])[1] > yx[d]) for d, s in enumerate(slc[-2:])]):
            found = True
            break
    if found:
        return i

