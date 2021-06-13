"""
Reads shapefiles, feature classes, table into a spatial dataframe
"""
from __future__ import print_function
from __future__ import division
import os
import six
import copy
import logging
import tempfile
from warnings import warn
import numpy as np
import pandas as pd
from six import iteritems, integer_types
from datetime import datetime
from urllib import request
from ..utils import NUMERIC_TYPES, STRING_TYPES, DATETIME_TYPES
from ..utils import sanitize_field_name
from arcgis.geometry import _types
try:
    import arcpy
    from arcpy import da
    HASARCPY = True
except:
    HASARCPY = False
try:
    import shapefile
    HASPYSHP = True
except:
    HASPYSHP = False
try:
    import fiona
    HASFIONA = True
except:
    HASFIONA = False

_log=logging.getLogger(__name__)

def _from_xy(df, x_column, y_column, sr=None):
    """

    """
    from arcgis.geometry import SpatialReference, Geometry
    from arcgis.features import SpatialDataFrame
    if sr is None:
        sr = SpatialReference({'wkid' : 4326})
    if not isinstance(sr, SpatialReference):
        if isinstance(sr, dict):
            sr = SpatialReference(sr)
        elif isinstance(sr, int):
            sr = SpatialReference({'wkid' : sr})
        elif isinstance(sr, str):
            sr = SpatialReference({'wkt' : sr})
    geoms = []
    for idx, row in df.iterrows():
        geoms.append(
            Geometry({'x' : row[x_column], 'y' : row[y_column],
             'spatialReference' : sr})
        )
    df['SHAPE'] = geoms
    return SpatialDataFrame(data=df, sr=sr)

def _pyshp_to_shapefile(df, out_path, out_name):
    """
    Saves a SpatialDataFrame to a Shapefile using pyshp

    :Parameters:
     :df: spatial dataframe
     :out_path: folder location to save the data
     :out_name: name of the shapefile
    :Output:
     path to the shapefile or None if pyshp isn't installed or
     spatial dataframe does not have a geometry column.
    """
    from .....geometry._types import Geometry
    if HASPYSHP:
        GEOMTYPELOOKUP = {
            "Polygon" : shapefile.POLYGON,
            "Point" : shapefile.POINT,
            "Polyline" : shapefile.POLYLINE,
            'null' : shapefile.NULL
        }
        if os.path.isdir(out_path) == False:
            os.makedirs(out_path)
        out_fc = os.path.join(out_path, out_name)
        if out_fc.lower().endswith('.shp') == False:
            out_fc += ".shp"
        geom_field = df.geometry.name
        if geom_field is None:
            return
        geom_type = "null"
        idx = df[geom_field].first_valid_index()
        if idx > -1:
            geom_type = df.loc[idx][geom_field].type
        shpfile = shapefile.Writer(GEOMTYPELOOKUP[geom_type])
        shpfile.autoBalance = 1
        row_cols = []
        for c in df.columns:
            idx = df[c].first_valid_index()
            if idx > -1:
                if isinstance(df[c].loc[idx],
                              Geometry):
                    geom_field = (c, "GEOMETRY")
                else:
                    row_cols.append(c)
                    if isinstance(df[c].loc[idx], six.string_types):
                        shpfile.field(name=c, size=255)
                    elif isinstance(df[c].loc[idx], six.integer_types):
                        shpfile.field(name=c, fieldType="N", size=5)
                    elif isinstance(df[c].loc[idx], (np.int, np.int32, np.int64)):
                        shpfile.field(name=c, fieldType="N", size=10)
                    elif isinstance(df[c].loc[idx], (np.float, np.float64)):
                        shpfile.field(name=c, fieldType="F", size=19, decimal=11)
                    elif isinstance(df[c].loc[idx], (datetime, np.datetime64)):
                        shpfile.field(name=c, fieldType="D", size=8)
                    elif isinstance(df[c].loc[idx], (bool, np.bool)):
                        shpfile.field(name=c, fieldType="L", size=1)
            del c
            del idx
        for idx, row in df.iterrows():
            geom = row[df.geometry.name]
            #del row[df.geometry.name]
            if geom.type == "Polygon":
                shpfile.poly(geom['rings'])
            elif geom.type == "Polyline":
                shpfile.line(geom['paths'])
            elif geom.type == "Point":
                shpfile.point(x=geom.x, y=geom.y)
            else:
                shpfile.null()
            shpfile.record(*row[row_cols].tolist())
            del idx
            del row
            del geom
        shpfile.save(out_fc)


        # create the PRJ file
        try:
            wkid = df.sr['wkid']
            try:
                wkid = df.sr['latestWkid']
            except:
                pass # try and use wkid instead

            prj_filename = out_fc.replace('.shp', '.prj')

            url = 'http://epsg.io/{}.esriwkt'.format(wkid)

            opener = request.build_opener()
            opener.addheaders = [('User-Agent', 'geosaurus')]
            resp = opener.open(url)

            wkt = resp.read().decode('utf-8')
            if len(wkt) > 0:
                prj = open(prj_filename, "w")
                prj.write(wkt)
                prj.close()
        except:
            # Unable to write PRJ file.
            pass

        del shpfile
        return out_fc
    return None
#--------------------------------------------------------------------------
def from_featureclass(filename, **kwargs):
    """
    Returns a GeoDataFrame from a feature class.
    Inputs:
     filename: full path to the feature class
    Optional Parameters:
     sql_clause: sql clause to parse data down
     where_clause: where statement
     sr: spatial reference object
     fields: list of fields to extract from the table
    """
    from .. import SpatialDataFrame
    from arcgis.geometry import _types
    if HASARCPY:
        sql_clause = kwargs.pop('sql_clause', (None,None))
        where_clause = kwargs.pop('where_clause', None)
        sr = kwargs.pop('sr', arcpy.Describe(filename).spatialReference or arcpy.SpatialReference(4326))
        fields = kwargs.pop('fields', None)
        desc = arcpy.Describe(filename)
        if not fields:
            fields = [field.name for field in arcpy.ListFields(filename) \
                      if field.type not in ['Geometry']]

            if hasattr(desc, 'areaFieldName'):
                afn = desc.areaFieldName
                if afn in fields:
                    fields.remove(afn)
            if hasattr(desc, 'lengthFieldName'):
                lfn = desc.lengthFieldName
                if lfn in fields:
                    fields.remove(lfn)
        geom_fields = fields + ['SHAPE@']
        flds = fields + ['SHAPE']
        vals = []
        geoms = []
        geom_idx = flds.index('SHAPE')
        shape_type = desc.shapeType
        default_polygon = _types.Geometry(arcpy.Polygon(arcpy.Array([arcpy.Point(0,0)]* 3)))
        default_polyline = _types.Geometry(arcpy.Polyline(arcpy.Array([arcpy.Point(0,0)]* 2)))
        default_point = _types.Geometry(arcpy.PointGeometry(arcpy.Point()))
        default_multipoint = _types.Geometry(arcpy.Multipoint(arcpy.Array([arcpy.Point()])))
        with arcpy.da.SearchCursor(filename,
                                   field_names=geom_fields,
                                   where_clause=where_clause,
                                   sql_clause=sql_clause,
                                   spatial_reference=sr) as rows:

            for row in rows:
                row = list(row)
                # Prevent curves/arcs
                if row[geom_idx] is None:
                    row.pop(geom_idx)
                    g = {}
                elif row[geom_idx].type in ['polyline', 'polygon']:
                    g = _types.Geometry(row.pop(geom_idx).generalize(0))
                else:
                    g = _types.Geometry(row.pop(geom_idx))
                if g == {}:
                    if shape_type.lower() == 'point':
                        g = default_point
                    elif shape_type.lower() == 'polygon':
                        g = default_polygon
                    elif shape_type.lower() == 'polyline':
                        g = default_point
                    elif shape_type.lower() == 'multipoint':
                        g = default_multipoint
                geoms.append(g)
                vals.append(row)
                del row
            del rows
        df = pd.DataFrame(data=vals, columns=fields)
        sdf = SpatialDataFrame(data=df, geometry=geoms)
        sdf.reset_index(drop=True, inplace=True)
        del df
        if sdf.sr is None:
            if sr is not None:
                sdf.sr = sr
            else:
                sdf.sr = sdf.geometry[sdf.geometry.first_valid_index()].spatialReference
        return sdf
    elif HASARCPY == False and \
         HASPYSHP == True and\
         filename.lower().find('.shp') > -1:
        geoms = []
        records = []
        reader = shapefile.Reader(filename)
        fields = [field[0] for field in reader.fields if field[0] != 'DeletionFlag']
        for r in reader.shapeRecords():
            atr = dict(zip(fields, r.record))
            g = r.shape.__geo_interface__
            g = _geojson_to_esrijson(g)
            geom = _types.Geometry(g)
            atr['SHAPE'] = geom
            records.append(atr)
            del atr
            del r, g
            del geom
        sdf = SpatialDataFrame(records)
        sdf.set_geometry(col='SHAPE')
        sdf.reset_index(inplace=True)
        return sdf
    elif HASARCPY == False and \
         HASPYSHP == False and \
         HASFIONA == True and \
         (filename.lower().find('.shp') > -1 or \
          os.path.dirname(filename).lower().find('.gdb') > -1):
        is_gdb = os.path.dirname(filename).lower().find('.gdb') > -1
        if is_gdb:
            with fiona.drivers():
                from arcgis.geometry import _types
                fp = os.path.dirname(filename)
                fn = os.path.basename(filename)
                geoms = []
                atts = []
                with fiona.open(path=fp, layer=fn) as source:
                    meta = source.meta
                    cols = list(source.schema['properties'].keys())
                    for idx, row in source.items():
                        geoms.append(_types.Geometry(row['geometry']))
                        atts.append(list(row['properties'].values()))
                        del idx, row
                    df = pd.DataFrame(data=atts, columns=cols)
                    return SpatialDataFrame(data=df, geometry=geoms)
        else:
            with fiona.drivers():
                from arcgis.geometry import _types
                geoms = []
                atts = []
                with fiona.open(path=filename) as source:
                    meta = source.meta
                    cols = list(source.schema['properties'].keys())
                    for idx, row in source.items():
                        geoms.append(_types.Geometry(row['geometry']))
                        atts.append(list(row['properties'].values()))
                        del idx, row
                    df = pd.DataFrame(data=atts, columns=cols)
                    return SpatialDataFrame(data=df, geometry=geoms)
    return
#--------------------------------------------------------------------------
def _arcpy_to_featureclass(df, out_name, out_location=None,
                           overwrite=True, out_sr=None,
                           skip_invalid=True):
    """
    """
    import arcgis
    import numpy as np
    import datetime
    from arcpy import da
    from arcgis.features import SpatialDataFrame

    gtype = df.geometry_type.upper()
    gname = df.geometry.name
    df = df.copy()

    if overwrite and \
       arcpy.Exists(os.path.join(out_location, out_name)):
        arcpy.Delete_management(os.path.join(out_location, out_name))
    elif overwrite == False and \
        arcpy.Exists(os.path.join(out_location, out_name)):
        raise Exception("Dataset exists, please provide a new out_name or location.")

    if out_sr is None:
        try:
            if isinstance(df.sr, dict):
                sr = arcgis.geometry.SpatialReference(df.sr).as_arcpy
            elif isinstance(df.sr, arcgis.geometry.SpatialReference):
                sr = df.sr.as_arcpy
        except:
            sr = arcpy.SpatialReference(4326)
    else:
        if isinstance(df.sr, dict):
            sr = arcgis.geometry.SpatialReference(df.sr).as_arcpy
        elif isinstance(df.sr, arcgis.geometry.SpatialReference):
            sr = df.sr.as_arcpy
        elif isinstance(out_sr, arcpy.SpatialReference):
            sr = out_sr
    fc = arcpy.CreateFeatureclass_management(out_path=out_location,
                                             out_name=out_name,
                                             geometry_type=gtype.upper(),
                                             spatial_reference=sr)[0]
    df['JOIN_ID_FIELD_DROP'] = df.index.tolist()
    flds = df.columns.tolist()

    flds.pop(flds.index(gname))
    flds_lower = [f.lower() for f in flds]
    for f in ['objectid', 'oid', 'fid']:
        if f in flds_lower:
            idx = flds_lower.index(f)
            flds.pop(idx)
            flds_lower.pop(idx)
            del idx
        del f
    array = [tuple(row) for row in df[flds].as_matrix()]
    geoms = df.geometry.as_arcpy.tolist()
    dtypes = []

    for idx, a in enumerate(array[0]):
        if isinstance(a,
                      STRING_TYPES):
            dtypes.append((flds[idx], '<U%s' %  df[flds[idx]].map(len).max()))
        elif flds[idx].lower() in ['fid', 'oid', 'objectid']:
            dtypes.append((flds[idx], np.int32))
        elif isinstance(a,
                        (int, np.int32)):
            dtypes.append((flds[idx], np.int64))
        elif isinstance(a,
                        (float, np.float, np.float64)):
            dtypes.append((flds[idx], np.float64))
        elif isinstance(a,
                        DATETIME_TYPES):
            dtypes.append((flds[idx], '<M8[us]'))
        else:
            dtypes.append((flds[idx], type(a)))
        del idx, a

    array = np.array(array, dtype=dtypes)
    del dtypes, flds, flds_lower

    with da.InsertCursor(fc, ['SHAPE@']) as icur:
        for g in geoms:
            if skip_invalid:
                try:
                    icur.insertRow([g])
                except: pass
            else:
                icur.insertRow([g])
    desc = arcpy.Describe(fc)
    oidField = desc.oidFieldName
    del desc
    da.ExtendTable(in_table=fc, table_match_field=oidField,
                   in_array=array, array_match_field='JOIN_ID_FIELD_DROP',
                   append_only=False)
    del df['JOIN_ID_FIELD_DROP']
    return fc
#--------------------------------------------------------------------------
def to_featureclass(df, out_name, out_location=None,
                    overwrite=True, out_sr=None,
                    skip_invalid=True):
    """
    converts a SpatialDataFrame to a feature class

    Parameters:
     :out_location: path to the workspace
     :out_name: name of the output feature class table
     :overwrite: True, the data will be erased then replaced, else the
      table will be appended to an existing table.
     :out_sr: if set, the data will try to reproject itself
     :skip_invalid: if True, the cursor object will not raise an error on
      insertion of invalid data, if False, the first occurence of invalid
      data will raise an exception.
    Returns:
     path to the feature class
    """
    fc = None
    if HASARCPY:
        import arcgis
        cols = []
        dt_idx = []
        invalid_rows = []
        idx = 0
        max_length = None
        if out_location:
            if os.path.isdir(out_location) == False and \
               out_location.lower().endswith('.gdb'):
                out_location = arcpy.CreateFileGDB_management(out_folder_path=os.path.dirname(out_location),
                                                             out_name=os.path.basename(out_location))[0]
            elif os.path.isdir(out_location) == False and \
                 out_name.lower().endswith('.shp'):
                os.makedirs(out_location)
            elif os.path.isfile(out_location) == False and \
                 out_location.lower().endswith('.sde'):
                raise ValueError("The sde connection file does not exist")
        else:
            if out_name.lower().endswith('.shp'):
                out_location = tempfile.gettempdir()
            elif HASARCPY:
                out_location = arcpy.env.scratchGDB
            else:
                out_location = tempfile.gettempdir()
                out_name = out_name + ".shp"
        fc = os.path.join(out_location, out_name)
        df = df.copy() # create a copy so we don't modify the source data.
        if out_name.lower().endswith('.shp'):
            max_length = 10
        for col in df.columns:
            if col.lower() != 'shape':
                if df[col].dtype.type in NUMERIC_TYPES:
                    df[col] = df[col].fillna(0)
                elif df[col].dtype.type in DATETIME_TYPES:
                    dt_idx.append(col)
                else:
                    df.loc[df[col].isnull(), col] = ""
                idx += 1
                col = sanitize_field_name(s=col,
                                          length=max_length)
            cols.append(col)
            del col
        df.columns = cols

        if arcpy.Exists(fc) and \
           overwrite:
            arcpy.Delete_management(fc)
        if arcpy.Exists(fc) ==  False:
            sr = df.sr
            if df.sr is None:
                sr = df['SHAPE'].loc[df['SHAPE'].first_valid_index()].spatial_reference
                if isinstance(sr, dict) and \
                   'wkid' in sr:
                    sr = arcpy.SpatialReference(sr['wkid'])
                elif isinstance(sr, arcpy.SpatialReference):
                    sr = sr
                else:
                    sr = None
            elif df.sr:
                sr = _types.SpatialReference(df.sr).as_arcpy
            elif sr is None:
                sr = df['SHAPE'].loc[df['SHAPE'].first_valid_index()].spatial_reference
                if isinstance(sr, dict) and \
                               'wkid' in sr:
                    sr = arcpy.SpatialReference(sr['wkid'])
                elif isinstance(sr, arcpy.SpatialReference):
                    sr = sr
                else:
                    sr = None
            elif isinstance(sr, dict):
                sr = _types.SpatialReference(sr).as_arcpy
            elif isinstance(sr, _types.SpatialReference):
                sr = df.sr.as_arcpy

            fc = arcpy.CreateFeatureclass_management(out_path=out_location,
                                                     out_name=out_name,
                                                     geometry_type=df.geometry_type.upper(),
                                                     spatial_reference=sr)[0]
        desc = arcpy.Describe(fc)
        oidField = desc.oidFieldName
        col_insert = copy.copy(df.columns).tolist()
        if hasattr(desc, 'areaFieldName'):
            af = desc.areaFieldName.lower()
        else:
            af = None
        if hasattr(desc, 'lengthFieldName'):
            lf = desc.lengthFieldName.lower()
        else:
            lf = None
        col_insert = [f for f in col_insert if f.lower() not in ['oid', 'objectid', 'fid', desc.oidFieldName.lower(), af, lf]]
        df_cols = col_insert.copy()
        lower_col_names = [f.lower() for f in col_insert if f.lower() not in ['oid', 'objectid', 'fid']]
        idx_shp = None

        if oidField.lower() in lower_col_names:
            val = col_insert.pop(lower_col_names.index(oidField.lower()))
            del df[val]
            col_insert = copy.copy(df.columns).tolist()
            lower_col_names = [f.lower() for f in col_insert]
        if hasattr(desc, "areaFieldName") and \
           desc.areaFieldName.lower() in lower_col_names:
            val = col_insert.pop(lower_col_names.index(desc.areaFieldName.lower()))
            del df[val]
            col_insert = copy.copy(df.columns).tolist()
            lower_col_names = [f.lower() for f in col_insert]
        elif 'shape_area' in lower_col_names:
            val = col_insert.pop(lower_col_names.index('shape_area'))
            del df[val]
            col_insert = copy.copy(df.columns).tolist()
            lower_col_names = [f.lower() for f in col_insert]
        if hasattr(desc, "lengthFieldName") and \
           desc.lengthFieldName.lower() in lower_col_names:
            val = col_insert.pop(lower_col_names.index(desc.lengthFieldName.lower()))
            del df[val]
            col_insert = copy.copy(df.columns).tolist()
            lower_col_names = [f.lower() for f in col_insert]
        elif 'shape_length' in lower_col_names:
            val = col_insert.pop(lower_col_names.index('shape_length'))
            del df[val]
            col_insert = copy.copy(df.columns).tolist()
            lower_col_names = [f.lower() for f in col_insert]
        if "SHAPE" in df.columns:
            idx_shp = col_insert.index("SHAPE")
            col_insert[idx_shp] = "SHAPE@"
        existing_fields = [field.name.lower() for field in arcpy.ListFields(fc)]
        for col in col_insert:
            if col.lower() != 'shape@' and \
               col.lower() != 'shape' and \
               col.lower() not in existing_fields:
                try:
                    t = _infer_type(df, col)
                    if t == "TEXT" and out_name.lower().endswith('.shp') == False:
                        l = int(df[col].str.len().max()) or 0
                        if l < 255:
                            l = 255
                        arcpy.AddField_management(in_table=fc, field_name=col,
                                                  field_length=l,
                                                  field_type=_infer_type(df, col))
                    else:
                        arcpy.AddField_management(in_table=fc, field_name=col,
                                              field_type=t)
                except:
                    print('col %s' % col)
        dt_idx = [col_insert.index(col) for col in dt_idx if col in col_insert]
        icur = da.InsertCursor(fc, col_insert)
        for index, row in df[df_cols].iterrows():
            if len(dt_idx) > 0:
                row = row.tolist()
                for i in dt_idx:
                    row[i] = row[i].to_pydatetime()
                    del i
                try:
                    if idx_shp:
                        row[idx_shp] = row[idx_shp].as_arcpy
                    icur.insertRow(row)
                except:
                    invalid_rows.append(index)
                    if skip_invalid == False:
                        raise Exception("Invalid row detected at index: %s" % index)
            else:
                try:
                    row = row.tolist()
                    if isinstance(idx_shp, int):
                        row[idx_shp] = row[idx_shp].as_arcpy
                    icur.insertRow(row)
                except:
                    invalid_rows.append(index)
                    if skip_invalid == False:
                        raise Exception("Invalid row detected at index: %s" % index)

            del row
        del icur
        if len(invalid_rows) > 0:
            t = ",".join([str(r) for r in invalid_rows])
            _log.warning('The following rows could not be written to the table: %s' % t)
    elif HASARCPY == False and \
         HASPYSHP:
        return _pyshp_to_shapefile(df=df,
                                   out_path=out_location,
                                   out_name=out_name)
    else:
        raise Exception("Cannot Export the data without ArcPy or PyShp modules. "+ \
                        "Please install them and try again.")
    return fc
#--------------------------------------------------------------------------
def _infer_type(df, col):
    """
    internal function used to get the datatypes for the feature class if
    the dataframe's _field_reference is NULL or there is a column that does
    not have a dtype assigned to it.

    Input:
     dataframe - spatialdataframe object
    Ouput:
      field type name
    """
    nn = df[col].notnull()
    nn = list(df[nn].index)
    if len(nn) > 0:
        val = df[col][nn[0]]
        if isinstance(val, six.string_types):
            return "TEXT"
        elif isinstance(val, tuple(list(six.integer_types) + [np.int32])):
            return "INTEGER"
        elif isinstance(val, (float, np.int64 )):
            return "FLOAT"
        elif isinstance(val, datetime):
            return "DATE"
    return "TEXT"
#--------------------------------------------------------------------------
def _geojson_to_esrijson(geojson):
    """converts the geojson spec to esri json spec"""
    if geojson['type'] in ['Polygon', 'MultiPolygon']:
        return {
            'rings' : geojson['coordinates']
        }
    elif geojson['type'] == "Point":
        return {
            "x" : geojson['coordinates'][0],
            "y" : geojson['coordinates'][1]
        }
    elif geojson['type'] == "MultiPoint":
        return {
            "points" : geojson['coordinates'],
        }
    elif geojson['type'] in ['LineString', 'MultiLineString']:
        return {
            "paths" : geojson['coordinates'],
        }
    return geojson
#--------------------------------------------------------------------------
def _geometry_to_geojson(geom):
    """converts the esri json spec to geojson"""
    if 'rings' in geom and \
       len(geom['rings']) == 1:
        return {
            'type' : "Polygon",
            "coordinates" : geom['rings']
        }
    elif 'rings' in geom and \
       len(geom['rings']) > 1:
        return {
            'type' : "MultiPolygon",
            "coordinates" : geom['rings']
        }
    elif geom['type'] == "Point":
        return {
            "coordinates" : [geom['x'], geom['y']],
            "type" : "Point"
        }
    elif geom['type'] == "MultiPoint":
        return {
            "coordinates" : geom['points'],
            'type' : "MultiPoint"
        }
    elif geom['type'].lower() == "polyline" and \
         len(geom['paths']) <= 1:
        return {
            "coordinates" : geom['paths'],
            'type' : "LineString"
        }
    elif geom['type'].lower() == "polyline" and \
         len(geom['paths']) > 1:
        return {
            "coordinates" : geom['paths'],
            'type' : "MultiLineString"
        }
    return geom