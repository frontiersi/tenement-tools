#!/usr/bin/env python
###############################################################################
# $Id$
#
#  Project:  GDAL samples
#  Purpose:  Rename file
#  Author:   Even Rouault <even.rouault at spatialys.com>
#
###############################################################################
#  Copyright (c) 2017, Even Rouault <even.rouault at spatialys.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
###############################################################################

import sys

from osgeo import gdal


def Usage():
    print('Usage: gdal_mv source target')
    return -1


def gdal_mv(argv, progress=None):
    source = None
    target = None
    simulate = False

    argv = gdal.GeneralCmdLineProcessor(argv)
    if argv is None:
        return -1

    for i in range(1, len(argv)):
        if len(argv[i]) == 0:
            return Usage()

        if argv[i] == '-simulate':
            simulate = True
        elif argv[i][0] == '-':
            print('Unexpected option : %s' % argv[i])
            return Usage()
        elif source is None:
            source = argv[i]
        elif target is None:
            target = argv[i]
        else:
            print('Unexpected option : %s' % argv[i])
            return Usage()

    if source is None or target is None:
        return Usage()

    if simulate:
        print('gdal.Rename(%s, %s)' % source, target)
        ret = 0
    else:
        ret = gdal.Rename(source, target)
    if ret != 0:
        print('Rename failed')
    return ret


if __name__ == '__main__':
    sys.exit(gdal_mv(sys.argv))
