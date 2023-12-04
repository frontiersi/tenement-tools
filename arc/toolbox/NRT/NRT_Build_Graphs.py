
import arcpy
import xarray as xr
import numpy as np
import os
class NRT_Build_Graphs(object):
    def __init__(self):
        """
        Initialise tool.
        """

        # set tool name, description, options
        self.label = 'NRT Build Graphs'
        self.description = 'Builds google charts when ' \
                           'user clicks on monitoring area. ' \
                           'Not intended for manual use.'
        self.canRunInBackground = False

    def getParameterInfo(self):
        """
        Set up UI parameters / controls.
        """

        # input netcdf path
        par_in_path = arcpy.Parameter(
            displayName='Input NetCDF path',
            name='in_nc',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_path.value = ''

        # input area parameters
        par_in_params = arcpy.Parameter(
            displayName='Parameters',
            name='in_parameters',
            datatype='GPString',
            parameterType='Required',
            direction='Input',
            multiValue=False)
        par_in_params.value = ''

        # combine parameters
        parameters = [
            par_in_path,
            par_in_params
        ]

        return parameters

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """
        Executes the NRT Build Graphs module.
        """


        # set up non-chart html
        _html_overview = """
        <html>
            <head>
                <style type="text/css">
                    h4 {
                        margin-top: 0px;
                        margin-bottom: 0px;
                    }
                    p {
                        margin-top: 0px;
                        margin-bottom: 0px;
                    }
                </style>
            </head>
            <body>
                <center>
                    <h3>Area Overview</h3>
                </center>

                <h4>Area identifier</h4>
                <p>//data.OvrAreaId</p>
                <br />

                <h4>Satellite platform</h4>
                <p>//data.OvrPlatform</p>
                <br />

                <h4>Starting year of pre-impact period</h4>
                <p>//data.OvrSYear</p>
                <br />

                <h4>Minimum number of training dates</h4>
                <p>//data.OvrEYear</p>
                <br />

                <h4>Vegetation index</h4>
                <p>//data.OvrIndex</p>
                <br />

                <h4>Model Persistence</h4>
                <p>//data.OvrPersistence</p>
                <br />

                <h4>Rule 1: Minimum consequtives</h4>
                <p>//data.OvrRule1MinConseq</p>
                <br />

                <h4>Rule 1: Include Plateaus</h4>
                <p>//data.OvrRule1IncPlateaus</p>
                <br />

                <h4>Rule 2: Minimum Zone</h4>
                <p>//data.OvrRule2MinZone</p>
                <br />

                <h4>Rule 3: Number of Zones</h4>
                <p>//data.OvrRule3NumZones</p>
                <br />

                <h4>Ruleset</h4>
                <p>//data.OvrRuleset</p>
                <br />

                <h4>Alert via email</h4>
                <p>//data.OvrAlert</p>
                <br />

                <h4>Alert Method</h4>
                <p>//data.OvrMethod</p>
                <br />

                <h4>Alert direction</h4>
                <p>//data.OvrDirection</p>
                <br />

                <h4>User email</h4>
                <p>//data.OvrEmail</p>
                <br />

                <h4>Ignore during monitoring</h4>
                <p>//data.OvrIgnore</p>
                <br />
            </body>
        </html>
        """

        # set up full veg html template
        _html_full_veg = """
        <html>
          <head>
            <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
            <script type="text/javascript">google.charts.load('current', {'packages': ['corechart', 'line']});

              google.charts.setOnLoadCallback(drawChart);

              function drawChart() {

                var data = new google.visualization.DataTable();
                data.addColumn('string', 'Date');
                data.addColumn('number', 'Vegetation (Raw)');
                data.addColumn('number', 'Vegetation (Smooth)');

                //data.addRows

                var options = {
                  legend: 'none',
                  chartArea: {
                    width: '100%',
                    height: '100%',
                    top: 20,
                    bottom: 100,
                    left: 75,
                    right: 25
                  },
                  hAxis: {
                    title: 'Date',
                    textStyle: {
                      fontSize: '9'
                    },
                    slantedText: true,
                    slantedTextAngle: 90
                  },
                  vAxis: {
                    title: 'Health (Median)',
                    textStyle: {
                      fontSize: '9'
                    },
                  },
                  series: {
                    0: {
                      color: 'grey',
                      lineWidth: 1,
                      enableInteractivity: false
                    },
                    1: {
                      color: 'green'
                    }
                  }
                };

                var chart = new google.visualization.LineChart(document.getElementById('line_chart'));
                chart.draw(data, options);
              }
            </script>
          </head>
          <body>
            <center>
              <h3>Vegetation History (Full)</h3>
              <div id="line_chart" style="width: 100%; height: 90%"></div>
            </center>
          </body>
        </html>
        """

        # set up sub veg html template
        _html_sub_veg = """
        <html>
          <head>
            <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
            <script type="text/javascript">google.charts.load('current', {'packages': ['corechart', 'line']});

              google.charts.setOnLoadCallback(drawChart);

              function drawChart() {

                var data = new google.visualization.DataTable();
                data.addColumn('string', 'Date');
                data.addColumn('number', 'Vegetation (Raw)');
                data.addColumn('number', 'Vegetation (Smooth)');

                //data.addRows

                var options = {
                  legend: 'none',
                  chartArea: {
                    width: '100%',
                    height: '100%',
                    top: 20,
                    bottom: 100,
                    left: 75,
                    right: 25
                  },
                  hAxis: {
                    title: 'Date',
                    textStyle: {
                      fontSize: '9'
                    },
                    slantedText: true,
                    slantedTextAngle: 90
                  },
                  vAxis: {
                    title: 'Health (Median)',
                    textStyle: {
                      fontSize: '9'
                    },
                  },
                  series: {
                    0: {
                      color: 'grey',
                      lineWidth: 1,
                      enableInteractivity: false
                    },
                    1: {
                      color: 'green'
                    }
                  }
                };

                var chart = new google.visualization.LineChart(document.getElementById('line_chart'));
                chart.draw(data, options);
              }

            </script>
          </head>

          <body>
            <center>
              <h3>Vegetation History (Analysis Only)</h3>
              <div id="line_chart" style="width: 100%; height: 90%"></div>
            </center>
          </body>

        </html>
        """

        # set up sub change html template
        _html_sub_change = """
        <html>
          <head>
            <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
            <script type="text/javascript">google.charts.load('current', {'packages': ['corechart', 'line']});

              google.charts.setOnLoadCallback(drawChart);

              function drawChart() {

                var data = new google.visualization.DataTable();
                data.addColumn('string', 'Date');
                data.addColumn('number', 'Change');
                data.addColumn('number', 'Alert');

                //data.addRows

                var options = {
                  legend: 'none',
                  chartArea: {
                    width: '100%',
                    height: '100%',
                    top: 20,
                    bottom: 100,
                    left: 75,
                    right: 25
                  },
                  hAxis: {
                    title: 'Date',
                    textStyle: {
                      fontSize: '9'
                    },
                    slantedText: true,
                    slantedTextAngle: 90
                  },
                  vAxis: {
                    title: 'Change Deviation',
                    textStyle: {
                      fontSize: '9'
                    },
                  },
                  series: {
                    0: {
                      color: 'red',
                      //lineWidth: 1,
                      //enableInteractivity: false
                    },
                    1: {
                        color: 'maroon',
                        lineWidth: 0,
                        pointSize: 5
                    }
                  }
                };

                var chart = new google.visualization.LineChart(document.getElementById('line_chart'));
                chart.draw(data, options);
              }

            </script>
          </head>

          <body>
            <center>
              <h3>Change Deviation & Alerts</h3>
              <div id="line_chart" style="width: 100%; height: 90%"></div>
            </center>
          </body>

        </html>
        """

        # set up sub zone html template
        _html_sub_zone = """
        <html>
            <head>
                <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
                <script type="text/javascript">
                    google.charts.load('current', { packages: ['corechart', 'bar']});
                    google.charts.setOnLoadCallback(drawChart);

                    function drawChart() {

                        //data.addRows

                        var options = {
                            legend: {
                                position: 'none',
                            },
                            chartArea: {
                                width: '100%', 
                                height: '100%',
                                top: 20,
                                bottom: 100,
                                left: 75,
                                right: 25
                                },
                            hAxis: {
                                title: 'Date',
                                textStyle: {fontSize: '9'},
                                slantedText: true, 
                                slantedTextAngle: 90
                            },
                            vAxis: {
                                title: 'Zone',
                                textStyle: {fontSize: '9'},
                            },
                        };

                        var chart = new google.visualization.ColumnChart(document.getElementById('column_chart'));
                        chart.draw(data, options);
                    }
                </script>
            </head>
            <body>
                <center>
                    <h3>Zone & Alert History</h3>
                    <div id="column_chart" style="width: 100%; height: 90%"></div>
                </center>

            </body>
        </html>
        """

        # set up legend html template
        _html_legend = """
        <html>
          <head>
            <style>
              td, th {
                border: 1px solid transparent;
                text-align: left;
                padding: 0px;
              }
            </style>
          </head>

          <body>
            <center>
              <h3>Zone Legend</h3>
            </center>

            <table style="width: 100%;">
              <colgroup>
                <col span="1" style="width: 15%;">
                <col span="1" style="width: 15%;">
                <col span="1" style="width: 70%;">
              </colgroup>
              <tr>
                <th>Symbology</th>
                <th>Zone</th>
                <th>Description</th>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #FF7F7F"></div>
                  </div>
                </td>
                <td>-11</td>
                <td>Change deviation is below -19. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #FFA77F"></div>
                  </div>
                </td>
                <td>-10</td>
                <td>Change deviation is between -17 and -19. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #FFD37F"></div>
                  </div>
                </td>
                <td>-9</td>
                <td>Change deviation is between -15 and -17. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #FFFF73"></div>
                  </div>
                </td>
                <td>-8</td>
                <td>Change deviation is between -13 and -15. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #D1FF73"></div>
                  </div>
                </td>
                <td>-7</td>
                <td>Change deviation is between -11 and -13. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #A3FF73"></div>
                  </div>
                </td>
                <td>-6</td>
                <td>Change deviation is between -9 and -11. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #73FFDF"></div>
                  </div>
                </td>
                <td>-5</td>
                <td>Change deviation is between -7 and -9. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #73DFFF"></div>
                  </div>
                </td>
                <td>-4</td>
                <td>Change deviation is between -5 and -7. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #73B2FF"></div>
                  </div>
                </td>
                <td>-3</td>
                <td>Change deviation is between -3 and -5. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #DF73FF"></div>
                  </div>
                </td>
                <td>-2</td>
                <td>Change deviation is between -1 and -3. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid red; background-color: #FF73DF"></div>
                  </div>
                </td>
                <td>-1</td>
                <td>Change deviation is between 0 and -1. Decline.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid black; background-color: white"></div>
                  </div>
                </td>
                <td>0</td>
                <td>No change in either direction.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #FF73DF"></div>
                  </div>
                </td>
                <td>1</td>
                <td>Change deviation is between 0 and 1. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #DF73FF"></div>
                  </div>
                </td>
                <td>2</td>
                <td>Change deviation is between 1 and 3. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #73B2FF"></div>
                  </div>
                </td>
                <td>3</td>
                <td>Change deviation is between 3 and 5. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #73DFFF"></div>
                  </div>
                </td>
                <td>4</td>
                <td>Change deviation is between 5 and 7. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #73FFDF"></div>
                  </div>
                </td>
                <td>5</td>
                <td>Change deviation is between 7 and 9. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #A3FF73"></div>
                  </div>
                </td>
                <td>6</td>
                <td>Change deviation is between 9 and 11. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #D1FF73"></div>
                  </div>
                </td>
                <td>7</td>
                <td>Change deviation is between 11 and 13. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #FFFF73"></div>
                  </div>
                </td>
                <td>8</td>
                <td>Change deviation is between 13 and 15. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #FFD37F"></div>
                  </div>
                </td>
                <td>9</td>
                <td>Change deviation is between 15 and 17. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #FFA77F"></div>
                  </div>
                </td>
                <td>10</td>
                <td>Change deviation is between 17 and 19. Growth.</td>
              </tr>
              <tr>
                <td>
                  <div>
                    <div style="width: 50px; height: 15px; border: 2px solid blue; background-color: #FF7F7F"></div>
                  </div>
                </td>
                <td>11</td>
                <td>Change deviation is above 19. Growth.</td>
              </tr>
            </table>
          </body>
        </html>
        """

        # grab parameter values
        in_path = parameters[0].value  # filepath to selected netcdf
        in_params = parameters[1].value  # string of area params seperated by ";"

        # # # # #
        #  prepare overview data html

        try:
            # unpack parameters
            params = in_params.split(';')

            # set over html values
            _html_overview = _html_overview.replace('//data.OvrAreaId', params[0])
            _html_overview = _html_overview.replace('//data.OvrPlatform', params[1])
            _html_overview = _html_overview.replace('//data.OvrSYear', params[2])
            _html_overview = _html_overview.replace('//data.OvrEYear', params[3])
            _html_overview = _html_overview.replace('//data.OvrIndex', params[4])
            _html_overview = _html_overview.replace('//data.OvrPersistence', params[5])
            _html_overview = _html_overview.replace('//data.OvrRule1MinConseq', params[6])
            _html_overview = _html_overview.replace('//data.OvrRule1IncPlateaus', params[7])
            _html_overview = _html_overview.replace('//data.OvrRule2MinZone', params[8])
            _html_overview = _html_overview.replace('//data.OvrRule3NumZones', params[9])
            _html_overview = _html_overview.replace('//data.OvrRuleset', params[10])
            _html_overview = _html_overview.replace('//data.OvrAlert', params[11])
            _html_overview = _html_overview.replace('//data.OvrMethod', params[12])
            _html_overview = _html_overview.replace('//data.OvrDirection', params[13])
            _html_overview = _html_overview.replace('//data.OvrEmail', params[14])
            html = _html_overview.replace('//data.OvrIgnore', params[15])
        except:
            html = '<h3>Could not generate overview information.</h3>'

        # add to output
        arcpy.AddMessage(html)

        # # # # #
        # check and load input netcdf at path

        try:
            # check if input path is valid
            if in_path is None or in_path == '':
                raise
            elif not os.path.exists(in_path):
                raise

            # safe open current dataset
            with xr.open_dataset(in_path) as ds:
                ds.load()

                # check if dataset is valid
            if 'time' not in ds:
                raise
            elif len(ds['time']) == 0:
                raise
            elif ds.to_array().isnull().all():
                raise

            # remove last date (-1 when slice)
            ds = ds.isel(time=slice(0, -1))
        except:
            pass  # proceed, causing error messages below

        # # # # #
        # prepare full vegetation history chart

        try:
            # unpack full time, veg raw, veg clean values
            ds_full = ds[['veg_idx', 'veg_clean']]
            dts = ds_full['time'].dt.strftime('%Y-%m-%d').values
            raw = ds_full['veg_idx'].values
            cln = ds_full['veg_clean'].values

            # prepare google chart data
            data = []
            for i in range(len(dts)):
                data.append([dts[i], raw[i], cln[i]])

            # replace nan with null, if exists
            data = str(data).replace('nan', 'null')

            # construct and relay full veg line chart html
            data = "data.addRows(" + data + ");"
            html = _html_full_veg.replace('//data.addRows', data)
        except:
            html = '<h3>Could not generate full vegetation history chart.</h3>'

        # add to output
        arcpy.AddMessage(html)

        # # # # #
        # prepare analysis vegetation history chart

        try:
            # drop dates < start year and get veg raw, clean
            ds_sub = ds.where(ds['time.year'] >= int(params[2]), drop=True)
            dts = ds_sub['time'].dt.strftime('%Y-%m-%d').values
            raw = ds_sub['veg_idx'].values
            cln = ds_sub['veg_clean'].values

            # prepare google chart data
            data = []
            for i in range(len(dts)):
                data.append([dts[i], raw[i], cln[i]])

            # replace nan with null, if exists
            data = str(data).replace('nan', 'null')

            # construct and relay full veg line chart html
            data = "data.addRows(" + data + ");"
            html = _html_sub_veg.replace('//data.addRows', data)
        except:
            html = '<h3>Could not generate analysis-only vegetation history chart.</h3>'

        # add to output
        arcpy.AddMessage(html)

        # # # # #
        # prepare change history chart

        try:
            # get method name and prepare vars
            method = params[12].lower()
            change_var = '{}_clean'.format(method)
            alert_var = '{}_alerts'.format(method)

            # get change
            chg = ds_sub[change_var].values

            # get where alerts exist
            alt = ds_sub[change_var].where(ds_sub[alert_var] == 1.0, np.nan)
            alt = alt.values

            # prepare google chart data
            data = []
            for i in range(len(dts)):
                data.append([dts[i], chg[i], alt[i]])

            # replace nan with null, if exists
            data = str(data).replace('nan', 'null')

            # construct and relay static change line chart html
            data = "data.addRows(" + data + ");"
            html = _html_sub_change.replace('//data.addRows', data)
        except:
            html = '<h3>Could not generate change history chart.</h3>'

        # add to output
        arcpy.AddMessage(html)

        # # # # #
        # prepare zone history chart

        try:
            # get method name and prepare vars
            method = params[12].lower()
            zone_var = '{}_zones'.format(method)
            alert_var = '{}_alerts'.format(method)

            # get zones where alerts exist
            zne = ds_sub[zone_var].where(ds_sub[alert_var] == 1.0, 0.0)
            zne = zne.values

            # prepare data statement and header row
            data_block = "var data = google.visualization.arrayToDataTable(["
            data_block += "['Date', 'Zone', {role: 'style'}],"

            # set zone colours
            cmap = {
                -12: "black",
                -11: "#FF7F7F",
                -10: "#FFA77F",
                -9: "#FFD37F",
                -8: "#FFFF73",
                -7: "#D1FF73",
                -6: "#A3FF73",
                -5: "#73FFDF",
                -4: "#73DFFF",
                -3: "#73B2FF",
                -2: "#DF73FF",
                -1: "#FF73DF",
                0: "#FFFFFF",
                1: "#FF73DF",
                2: "#DF73FF",
                3: "#73B2FF",
                4: "#73DFFF",
                5: "#73FFDF",
                6: "#A3FF73",
                7: "#D1FF73",
                8: "#FFFF73",
                9: "#FFD37F",
                10: "#FFA77F",
                11: "#FF7F7F",
                12: "black"
            }

            # prepare google chart data
            data = []
            for i in range(len(dts)):
                data.append([dts[i], zne[i], cmap.get(zne[i])])

            # construct string data array
            data = ','.join([str(s) for s in data])

            # replace nan with null, if exists
            data = data.replace('nan', 'null')

            # finalise block
            data_block += data + "]);"

            # # prepare data
            html = _html_sub_zone.replace('//data.addRows', data_block)
        except:
            html = '<h3>Could not generate zone history chart.</h3>'

        # add to output
        arcpy.AddMessage(html)

        # finally, add static legend chart
        arcpy.AddMessage(_html_legend)

        return