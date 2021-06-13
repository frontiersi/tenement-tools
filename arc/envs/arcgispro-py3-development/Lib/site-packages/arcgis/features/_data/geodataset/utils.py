"""
    Common set of utilities to assist with operations.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import re
from .index.rtree import Rtree
from datetime import datetime as _datetime
import pandas as pd
import numpy as np

if [float(i) for i in pd.__version__.split('.')] < [1,0,0]:
    DATETIME_TYPES = (_datetime,
                      np.datetime64,
                      pd.datetime,
                      pd.DatetimeIndex)
else:
    DATETIME_TYPES = (_datetime,
                      np.datetime64)

NUMERIC_TYPES = tuple([int] + [
    np.int, np.int16,
    np.int32, np.integer,
    np.float, np.float32,
    np.float64, np.int8,
    np.int64, np.short])

STRING_TYPES = tuple([str] + \
    [str, np.str, np.unicode, chr])

# --------------------------------------------------------------------------
def chunks(l, n):
    """yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
#--------------------------------------------------------------------------
def sanitize_field_name(s, length=None, sub_value=None):
    """
    Modifies the string by replacing special characters by another value.
    It can also shorten a string if length is specified.

    Parmaters:
     :s: string value
     :length: optional integer value must be > 0
     :sub_value: optional replacement value instead of empty string

    Returns:
     string

    Usage:
    >>> sv = 'how much for the doggie in the window? $20.99?'
    >>> print (sanitize_field_name(sv))
    howmuchforthedoggieinthewindow2099
    """
    if sub_value is None:
        sub_value = ""
    s = re.sub('\W+', sub_value, s)
    if isinstance(length, int) and \
       length > 0 and \
       len(s) > length:
        s = s[:length]
    return s
