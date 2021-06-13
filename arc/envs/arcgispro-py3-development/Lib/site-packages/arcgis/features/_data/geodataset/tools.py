from arcgis.geometry._types import Geometry

def spatial_join(df1, df2, left_tag="_left", right_tag="_right", keep_all=True):
    """
    Joins two spatailly enabled dataframes based on spatial location based
    on if the two geometries are intersected.

    Parameters:
      :df1: left spatial dataframe
      :df2: right spatial dataframe
      :left_tag: if the same column is in the left and right dataframe,
       this will append that string value to the field
      :right_tag: if the same column is in the left and right dataframe,
       this will append that string value to the field
      :keep_all: if set to true all df1 will be kept regardless of spatial
       matches
    :output:
      Spatial Dataframe
    """
    import numpy as np
    import pandas as pd
    from arcgis.features import SpatialDataFrame
    if not isinstance(df1, SpatialDataFrame):
        raise ValueError("df1 must be a spatial dataframe")
    if not isinstance(df2, SpatialDataFrame):
        raise ValueError("df2 must be a spatial dataframe")
    right_index = df2.sindex
    join_idx = []
    if not df1.geometry is None:
        geom_field = df1.geometry.name
    else:
        raise ValueError("df1 is missing a geometry column")
    if not df2.geometry is None:
        geom_field2 = df2.geometry.name
    else:
        raise ValueError("df2 is missing a geometry column")
    for idx, row in df1.iterrows():
        geom = row[geom_field]
        if isinstance(geom.extent, tuple):
            ext = (geom.extent[0], geom.extent[1], geom.extent[2], geom.extent[3])
        else:
            ext = (geom.extent.XMin, geom.exten.YMin, geom.extent.XMax, geom.extent.YMax)
        select_idx = right_index.intersect(ext)
        if len(select_idx) > 0:
            sub = df2.loc[select_idx]
            res = sub[sub.disjoint(geom) == False]
            if len(res) > 0:
                for idx2, row2 in res.iterrows():
                    join_idx.append([idx, idx2])
                    del idx2, row2
            elif len(res) == 0 and keep_all:
                join_idx.append([idx, None])
            del sub, res
        elif len(select_idx) == 0 and \
             keep_all:
            join_idx.append([idx, None])
        del geom
        del ext
        del select_idx
        del idx
    join_field_names = ["TARGET_OID",
                        "JOIN_OID"]
    df2 = df2.copy()
    del df2[df2.geometry.name]
    join_df = pd.DataFrame(data=join_idx, columns=join_field_names)
    join_df = join_df.merge(df1,
                            left_on=join_field_names[0],
                            right_index=True,
                            how='left',
                            suffixes=(left_tag,
                                      right_tag))
    join_df = join_df.merge(df2,
                            left_on=join_field_names[1],
                            right_index=True, how='left',
                            suffixes=(left_tag, right_tag),
                            copy=True)
    join_df = SpatialDataFrame(join_df)
    join_df.geometry = join_df[df1.geometry.name]
    del join_idx
    join_df.reset_index(drop=True, inplace=True)
    return join_df
