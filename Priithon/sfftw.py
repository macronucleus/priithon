# This file was automatically generated by SWIG (http://www.swig.org).
# Version 1.3.31
#
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _sfftw
import new
new_instancemethod = new.instancemethod
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'PySwigObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


FFTW_FORWARD = _sfftw.FFTW_FORWARD
FFTW_BACKWARD = _sfftw.FFTW_BACKWARD
FFTW_SUCCESS = _sfftw.FFTW_SUCCESS
FFTW_FAILURE = _sfftw.FFTW_FAILURE
FFTW_NOTW = _sfftw.FFTW_NOTW
FFTW_TWIDDLE = _sfftw.FFTW_TWIDDLE
FFTW_GENERIC = _sfftw.FFTW_GENERIC
FFTW_RADER = _sfftw.FFTW_RADER
FFTW_REAL2HC = _sfftw.FFTW_REAL2HC
FFTW_HC2REAL = _sfftw.FFTW_HC2REAL
FFTW_HC2HC = _sfftw.FFTW_HC2HC
FFTW_RGENERIC = _sfftw.FFTW_RGENERIC
FFTW_NORMAL_RECURSE = _sfftw.FFTW_NORMAL_RECURSE
FFTW_VECTOR_RECURSE = _sfftw.FFTW_VECTOR_RECURSE
FFTW_ESTIMATE = _sfftw.FFTW_ESTIMATE
FFTW_MEASURE = _sfftw.FFTW_MEASURE
FFTW_OUT_OF_PLACE = _sfftw.FFTW_OUT_OF_PLACE
FFTW_IN_PLACE = _sfftw.FFTW_IN_PLACE
FFTW_USE_WISDOM = _sfftw.FFTW_USE_WISDOM
FFTW_THREADSAFE = _sfftw.FFTW_THREADSAFE
FFTWND_FORCE_BUFFERED = _sfftw.FFTWND_FORCE_BUFFERED
FFTW_NO_VECTOR_RECURSE = _sfftw.FFTW_NO_VECTOR_RECURSE

def rfftw(*args):
  """
    rfftw(rfftw_plan plan, int howmany, fftw_real in, int istride, 
        int idist, fftw_real out, int ostride, 
        int odist)
    """
  return _sfftw.rfftw(*args)

def rfftw_one(*args):
  """rfftw_one(rfftw_plan plan, fftw_real in, fftw_real out)"""
  return _sfftw.rfftw_one(*args)

def rfftwnd_create_plan(*args):
  """rfftwnd_create_plan(int rank, int array00d, fftw_direction dir, int flags) -> rfftwnd_plan"""
  return _sfftw.rfftwnd_create_plan(*args)

def rfftwnd_destroy_plan(*args):
  """rfftwnd_destroy_plan(rfftwnd_plan plan)"""
  return _sfftw.rfftwnd_destroy_plan(*args)

def rfftwnd_one_real_to_complex(*args):
  """rfftwnd_one_real_to_complex(rfftwnd_plan p, fftw_real in, fftw_complex out)"""
  return _sfftw.rfftwnd_one_real_to_complex(*args)

def rfftwnd_one_complex_to_real(*args):
  """rfftwnd_one_complex_to_real(rfftwnd_plan p, fftw_complex in, fftw_real out)"""
  return _sfftw.rfftwnd_one_complex_to_real(*args)

def fftw_sizeof_fftw_real(*args):
  """fftw_sizeof_fftw_real() -> size_t"""
  return _sfftw.fftw_sizeof_fftw_real(*args)
class fftwnd_data(_object):
    """Proxy of C++ fftwnd_data class"""
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, fftwnd_data, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, fftwnd_data, name)
    __repr__ = _swig_repr
    __swig_setmethods__["is_in_place"] = _sfftw.fftwnd_data_is_in_place_set
    __swig_getmethods__["is_in_place"] = _sfftw.fftwnd_data_is_in_place_get
    if _newclass:is_in_place = _swig_property(_sfftw.fftwnd_data_is_in_place_get, _sfftw.fftwnd_data_is_in_place_set)
    __swig_setmethods__["rank"] = _sfftw.fftwnd_data_rank_set
    __swig_getmethods__["rank"] = _sfftw.fftwnd_data_rank_get
    if _newclass:rank = _swig_property(_sfftw.fftwnd_data_rank_get, _sfftw.fftwnd_data_rank_set)
    __swig_setmethods__["n"] = _sfftw.fftwnd_data_n_set
    __swig_getmethods__["n"] = _sfftw.fftwnd_data_n_get
    if _newclass:n = _swig_property(_sfftw.fftwnd_data_n_get, _sfftw.fftwnd_data_n_set)
    __swig_setmethods__["dir"] = _sfftw.fftwnd_data_dir_set
    __swig_getmethods__["dir"] = _sfftw.fftwnd_data_dir_get
    if _newclass:dir = _swig_property(_sfftw.fftwnd_data_dir_get, _sfftw.fftwnd_data_dir_set)
    __swig_setmethods__["n_before"] = _sfftw.fftwnd_data_n_before_set
    __swig_getmethods__["n_before"] = _sfftw.fftwnd_data_n_before_get
    if _newclass:n_before = _swig_property(_sfftw.fftwnd_data_n_before_get, _sfftw.fftwnd_data_n_before_set)
    __swig_setmethods__["n_after"] = _sfftw.fftwnd_data_n_after_set
    __swig_getmethods__["n_after"] = _sfftw.fftwnd_data_n_after_get
    if _newclass:n_after = _swig_property(_sfftw.fftwnd_data_n_after_get, _sfftw.fftwnd_data_n_after_set)
    __swig_setmethods__["plans"] = _sfftw.fftwnd_data_plans_set
    __swig_getmethods__["plans"] = _sfftw.fftwnd_data_plans_get
    if _newclass:plans = _swig_property(_sfftw.fftwnd_data_plans_get, _sfftw.fftwnd_data_plans_set)
    __swig_setmethods__["nbuffers"] = _sfftw.fftwnd_data_nbuffers_set
    __swig_getmethods__["nbuffers"] = _sfftw.fftwnd_data_nbuffers_get
    if _newclass:nbuffers = _swig_property(_sfftw.fftwnd_data_nbuffers_get, _sfftw.fftwnd_data_nbuffers_set)
    __swig_setmethods__["nwork"] = _sfftw.fftwnd_data_nwork_set
    __swig_getmethods__["nwork"] = _sfftw.fftwnd_data_nwork_get
    if _newclass:nwork = _swig_property(_sfftw.fftwnd_data_nwork_get, _sfftw.fftwnd_data_nwork_set)
    __swig_setmethods__["work"] = _sfftw.fftwnd_data_work_set
    __swig_getmethods__["work"] = _sfftw.fftwnd_data_work_get
    if _newclass:work = _swig_property(_sfftw.fftwnd_data_work_get, _sfftw.fftwnd_data_work_set)
    def __init__(self, *args): 
        """__init__(self) -> fftwnd_data"""
        this = _sfftw.new_fftwnd_data(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _sfftw.delete_fftwnd_data
    __del__ = lambda self : None;
fftwnd_data_swigregister = _sfftw.fftwnd_data_swigregister
fftwnd_data_swigregister(fftwnd_data)


def fftwnd_create_plan(*args):
  """fftwnd_create_plan(int rank, int array00d, fftw_direction dir, int flags) -> fftwnd_plan"""
  return _sfftw.fftwnd_create_plan(*args)

def fftwnd_destroy_plan(*args):
  """fftwnd_destroy_plan(fftwnd_plan plan)"""
  return _sfftw.fftwnd_destroy_plan(*args)

def fftwnd_one(*args):
  """fftwnd_one(fftwnd_plan p, fftw_complex in, fftw_complex out)"""
  return _sfftw.fftwnd_one(*args)


