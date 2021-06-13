from ._util import _to_datetime, _datetime2ole, _ole2datetime, _iso_to_datetime, _check_if_iso_format, _time_filter, _linear_regression, _harmonic_regression
import datetime as _datetime
import logging as _logging
_LOGGER = _logging.getLogger(__name__)


try:
    import numpy as _np
    import matplotlib.pyplot as _plt
    from matplotlib.pyplot import cm as _cm

except:
    raise
    
    

def temporal_profile(raster, points=[], time_field=None, variables=[],  bands=[0], time_extent=None, dimension=None, dimension_values=[], 
                     show_values=False, trend_type=None, trend_order=None, plot_properties={}):

    '''
    A temporal profile serves as a basic analysis tool for imagery data in a time series. 
    Visualizing change over time with the temporal profile allows trends to be displayed 
    and compared with variables, bands, or values from other dimensions simultaneously.

    Using the functionality in temporal profile charts, you can perform trend analysis, gain insight into 
    multidimensional raster data at given locations, and plot values that are changing over time 
    in the form of a line graph.

    Temporal profile charts can be used in various scientific applications involving time series 
    analysis of raster data, and the graphical output of results can be used directly as 
    input for strategy management and decision making.

    The x-axis of the temporal profile displays the time in continuous time intervals. The time field is 
    obtained from the timeInfo of the image service.
    
    The y-axis of the temporal profile displays the variable value.


    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    raster                                   Required Imagery Layer object.
    ------------------------------------     --------------------------------------------------------------------
    points                                   Required list of point Geometry objects. 
    ------------------------------------     --------------------------------------------------------------------
    time_field                               Required string. The time field that will be used for plotting 
                                             temporal profile.
                                             
                                             If not specified the time field is obtained from the timeInfo of 
                                             the image service.
    ------------------------------------     --------------------------------------------------------------------
    variables                                Required list of variable names. 
                                             For non multidimensional data, the variable would be name of the Sensor.
                                             To plot the graph against all sensors specify - "ALL_SENSORS" 
    ------------------------------------     --------------------------------------------------------------------
    bands                                    Optional list of band indices. By default takes the 
                                             first band (band index - 0). 
                                             For a multiband data, you can compare the time change of different 
                                             bands over different locations.
    ------------------------------------     --------------------------------------------------------------------
    time_extent                              Optional list of date time object. This represents the time extent
    ------------------------------------     --------------------------------------------------------------------
    dimension                                Optional list of dimension names. This option works specifically on 
                                             multidimensional data containing a time dimension and other dimensions.

                                             The temporal profile is created based on the specific values in other 
                                             dimensions, such as depth at the corresponding time value. For example, 
                                             soil moisture data usually includes both a time dimension and vertical 
                                             dimension below the earth's surface, resulting in a temporal profile 
                                             at 0.1, 0.2, and 0.3 meters below the ground.
    ------------------------------------     --------------------------------------------------------------------
    dimension_values                         Optional list of dimension values. This parameter can be used to specify
                                             the values of dimension parameter other than the time dimension (dimension
                                             name specified using dimension parameter)
    ------------------------------------     --------------------------------------------------------------------
    show_values                              Optional bool. Default False.
                                             Set this parameter to True to display the values at each point in the line graph.
    ------------------------------------     --------------------------------------------------------------------
    trend_type                               Optional string. Default None.
                                             Set the trend_type parameter eith with linear or harmonic to draw the trend line
                                             linear : Fits the pixel values for a variable along a linear trend line.
                                             harmonic : Fits the pixel values for a variable along a harmonic trend line.
    ------------------------------------     --------------------------------------------------------------------
    trend_order                              optional number. The frequency number to use in the trend fitting. 
                                             This parameter specifies the frequency of cycles in a year. 
                                             The default value is 1, or one harmonic cycle per year.

                                             This parameter is only included in the trend analysis for a harmonic regression.
    ------------------------------------     --------------------------------------------------------------------
    plot_properties                          Optional dict. This parameter can be used to set the figure 
                                             properties. These are the matplotlib.pyplot.figure() parameters and values
                                             specified in dict format.

                                             eg: {"figsize":(15,15)}
    ====================================     ====================================================================

    :return:
        None

    '''
    
    t1 = []

    if not isinstance(variables,list):
        variables= [variables]
    if not isinstance(points,list):
        points= [points]
    if not isinstance(bands,list):
        bands= [bands]

    if time_field is None:
        try:
            x_var = raster.properties.timeInfo['startTimeField']
        except:
            raise RuntimeError("Specify time_field to plot the temporal profile.")
    else:
        x_var = time_field

    if "hasMultidimensions" in raster.properties and \
            raster.properties['hasMultidimensions'] == True:
        
        mosaic_rule = {
                        "mosaicMethod" : "esriMosaicAttribute",
                        "ascending" : False,
                        "sortField": x_var,
                        "multidimensionalDefinition": [{
                            "variableName" : "",
                            "dimensionName" : x_var
                            }]
                    }
        if time_extent is not None:
            if isinstance(time_extent, _datetime.datetime):
                time_extent =[int(time_extent.timestamp() * 1000)]
            elif isinstance(time_extent, list):
                if isinstance(time_extent[0], _datetime.datetime) and isinstance(time_extent[1], _datetime.datetime):
                    time_extent = [int(time_extent[0].timestamp() * 1000),
                                                int(time_extent[1].timestamp() * 1000)]
            for index, each_elem in enumerate(mosaic_rule['multidimensionalDefinition']):
                if mosaic_rule['multidimensionalDefinition'][index]['dimensionName'] == x_var:
                    mosaic_rule['multidimensionalDefinition'][index]['values']=[time_extent]

        if dimension is not None and dimension_values is not None:
            if not isinstance(dimension_values,list):
                dimension_values = [dimension_values]
            mosaic_rule['multidimensionalDefinition'].append({
                            "variableName" : "",
                            "dimensionName" : dimension,
                        "values":dimension_values,
                        "isSlice":True})

        num_lines = len(dimension_values)*len(variables)*len(points)*len(bands)
        y=[[] for i in range(0, num_lines)]
        x=[[] for i in range(0, num_lines)]
        #x_var = raster.properties.timeInfo['startTimeField']

        if len(variables)==1:
            variable_unit = None
            for ele in raster.multidimensional_info['multidimensionalInfo']['variables']:
                if(ele['name']==variables[0]):
                    if "unit" in ele.keys():
                        variable_unit = ele['unit']
                        break
        res=[]
        t1=[]
        d1=[]
        xx=[]
        yy=[]
        for band in bands:
            for index, point in enumerate(points):
                for variable in variables:
                    for md_def in mosaic_rule['multidimensionalDefinition']:
                        md_def["variableName"]=variable

                    res=raster.get_samples(geometry=point, return_first_value_only=False, out_fields="*", mosaic_rule=mosaic_rule)

                    if dimension_values !=[]:
                        for dim_value in dimension_values:
                            for res_ele in res:
                                if(res_ele['attributes'][dimension]==dim_value):
                                    yy.append(res_ele['values'][band])
                                    xx.append(_to_datetime(res_ele['attributes'][x_var]))
                            d1.append({"yy":yy, "xx":xx, "dimension_value":dim_value})
                            yy=[]
                            xx=[]
                        

                    else:
                        for ele in res:
                            y.append(ele['values'][band])
                            x.append(_to_datetime(ele['attributes'][x_var]))

                    #if "bandNames" in raster.properties:
                    #        band = raster.properties.bandNames[band]

                    if dimension_values ==[]:
                        t1.append({"y":y,
                                    "x":x,
                                    "point": index,
                                    "variable":variable,
                                    "band":band})
                    else:
                        for ele in d1:
                            t1.append({"y":ele["yy"],
                                        "x":ele["xx"],
                                        "point": index,
                                        "variable":variable,
                                        "dimension_value":ele["dimension_value"],
                                        "dimension":dimension,
                                        "band":band})
                    x=[]
                    y=[]
                    d1=[]

        if plot_properties is None:
            plot_properties = {}
        if len(plot_properties)==0 or (len(plot_properties)>0 and "figsize" not in plot_properties.keys()):
            plot_properties.update({"figsize":(15,15)})
        if plot_properties is not None and isinstance(plot_properties,dict):
            #{"figsize":(20,10),"dpi":100,"facecolor":"yellow","edgecolor":"blue","linewidth":10.0,"frameon":False}
            _plt.figure(**plot_properties)
        _plt.xlabel(x_var)
        if len(variables)==1:
            if variable_unit is  not None:
                _plt.ylabel(variables[0] + ' (in '+variable_unit+')')
            else:
                _plt.ylabel(variables[0])
        else:
            _plt.ylabel("Values")

        title_string = "Change in"
        for ele in variables:
            title_string=title_string+" "+str(ele+",")
        title_string=title_string+ " over "+x_var
        if dimension is not None and dimension_values is not None:
            title_string = title_string+ ','' at '+ str(dimension)+' =  '+str(dimension_values)
        _plt.title(title_string)

        color=iter(_cm.rainbow(_np.linspace(0,1,len(t1))))
        for i in range(0,len(t1)):
            label_string =  "Location "+ str(t1[i]["point"])+"-"+ str(t1[i]["variable"])
            if "dimension" in t1[i].keys():
                label_string = label_string+"-"+t1[i]["dimension"]+"="+str(t1[i]["dimension_value"])
            if "band" in t1[i].keys():
                label_string = label_string+"-"+"band = "+str(t1[i]["band"])
            c=next(color)
            _plt.plot(t1[i]["x"],t1[i]["y"], c=c, label=label_string)
            _plt.scatter(t1[i]["x"],t1[i]["y"], c=[c])
            _plt.legend(loc='upper left')

        #for i in range(0,len(t1)):
        #    label_string =  "Location "+ str(t1[i]["point"])+"-"+ str(t1[i]["variable"])
        #    if "dimension" in t1[i].keys():
        #        label_string = label_string+"-"+t1[i]["dimension"]+"="+str(t1[i]["dimension_value"])
        #    plt.plot(t1[i]["x"],t1[i]["y"], label=label_string)
        #    plt.scatter(t1[i]["x"],t1[i]["y"])
        #    plt.legend(loc='upper left')
        #plt.gcf().autofmt_xdate()
            #print(t1[i]["x"]," < ", t1[i]["y"])

            if trend_type is not None:
                date_list=[]
                for date in t1[i]["x"]:
                    ole_date = _datetime2ole(date)
                    date_list.append(ole_date)
                sample_size = len(date_list)
                if (sample_size != len(t1[i]["y"])):
                    print("error")
                if trend_type.lower() == "linear":
                    x_trend, y_trend = _linear_regression(sample_size, date_list, t1[i]["x"], t1[i]["y"])
                elif trend_type.lower() == "harmonic":
                    if trend_order is None:
                        _LOGGER.warning("Trend line cannot be drawn. Please enter a trend order value from 1 to 3.")
                    if trend_order < 1:
                        trend_order = 1
                        _LOGGER.warning("Invalid Argument - trend order is less than 1. Setting trend order as 1 to plot the trend line")
                    if trend_order > 3:
                        trend_order = 3
                        _LOGGER.warning("Invalid Argument - trend order is greater than 3. Setting trend order as 3 to plot the trend line")
                    x_trend, y_trend = _harmonic_regression(sample_size, date_list, t1[i]["x"], t1[i]["y"], trend_order)
                _plt.plot(x_trend, y_trend,"--g")

            if show_values:
                for x,y in zip(t1[i]["x"],t1[i]["y"]):
                    label = "{:.2f}".format(y)
                    _plt.annotate(label, # this is the text
                                (x,y), # this is the point to label
                                textcoords="offset points", # how to position the text
                                xytext=(10,5), # distance from text to points (x,y)
                                ha='center') 

        _plt.show()
        #plt.legend()
    else:
        num_lines = len(points)*len(bands)
        #y=[[] for i in range(0, num_lines)]
        #x=[[] for i in range(0, num_lines)]
        t2=[]
        t1=[]
        xx=[]
        yy=[]
        d1=[]

        mosaic_rule = {
                "mosaicMethod" : "esriMosaicAttribute",
                "ascending" : False,
                "sortField": x_var
            }

        for index, point in enumerate(points):
            for variable in variables:
                t2 = raster.get_samples(geometry=point, return_first_value_only=False, out_fields="*" ,mosaic_rule = mosaic_rule)
                #print(t2)
                #x_var = raster.properties.timeInfo['startTimeField']
               
                for band in bands:
                    for element in t2:                    
                        if "attributes" in element:
                            if "SensorName" in element["attributes"].keys():
                                if variable.upper()=="ALL_SENSORS":
                                    if _time_filter(time_extent, _to_datetime(element["attributes"][x_var])) == True:
                                        yy.append(element['values'][band])
                                        xx.append(_to_datetime(element["attributes"][x_var]))
                                        xx, yy = zip(*sorted(zip(xx, yy)))
                                        xx=list(xx)
                                        yy=list(yy)
                                if element["attributes"]["SensorName"]==variable:
                                    if _time_filter(time_extent, _to_datetime(element["attributes"][x_var])) == True:
                                        yy.append(element['values'][band])
                                        xx.append(_to_datetime(element["attributes"][x_var]))
                                        xx, yy = zip(*sorted(zip(xx, yy)))
                                        xx=list(xx)
                                        yy=list(yy)

                    d1.append({"yy":yy, "xx":xx, "band":band})
                    yy=[]
                    xx=[]            
                for ele in d1:
                    t1.append({"y":ele["yy"],
                                "x":ele["xx"],
                                "point": index,
                                "variable" : variable,
                                "band":ele["band"]})
                d1=[]
        #print(t1)
        if plot_properties is None:
            plot_properties = {}
        if len(plot_properties)==0 or (len(plot_properties)>0 and "figsize" not in plot_properties.keys()):
            plot_properties.update({"figsize":(15,15)})
        if plot_properties is not None and isinstance(plot_properties,dict):
            #{"figsize":(20,10),"dpi":100,"facecolor":"yellow","edgecolor":"blue","linewidth":10.0,"frameon":False}
            _plt.figure(**plot_properties)
        #_plt.figure(figsize=(15,15))
        #_plt.figure()
        _plt.xlabel(x_var)
        _plt.ylabel(variable)
        if len(variables)==1:
            _plt.ylabel(variables[0])
        else:
            _plt.ylabel("Values")
        title_string = "Change in"
        for ele in variables:
            title_string=title_string+" "+str(ele+",")
        title_string=title_string+ " over "+x_var
        _plt.title(title_string)
        color=iter(_cm.rainbow(_np.linspace(0,1,len(t1))))
        for i in range(0,len(t1)):
            label_string =  "Location "+ str(t1[i]["point"])+"-"+ str(t1[i]["variable"])
            #label_string =  "Location "+ str(t1[i]["point"])+"-"
            if "band" in t1[i].keys():
                label_string = label_string+"-"+"band = "+str(t1[i]["band"])
            c=next(color)
            _plt.plot(t1[i]["x"],t1[i]["y"], c=c, label=label_string)
            _plt.scatter(t1[i]["x"],t1[i]["y"], c=[c])
            _plt.legend(loc='upper left')
            
            if show_values:
                for x,y in zip(t1[i]["x"],t1[i]["y"]):
                    label = "{:.2f}".format(y)
                    _plt.annotate(label, # this is the text
                                (x,y), # this is the point to label
                                textcoords="offset points", # how to position the text
                                xytext=(10,5), # distance from text to points (x,y)
                                ha='center') 
        #_plt.gcf().autofmt_xdate()
        _plt.show()

