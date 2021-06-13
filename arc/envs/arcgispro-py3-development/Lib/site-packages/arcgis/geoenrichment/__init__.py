"""
The arcgis.geoenrichment module  enables you to answer questions about locations that you can't answer with maps alone.

GeoEnrichment provides the ability to get facts about a location or area. Using GeoEnrichment, you can get information
about the people, places, and businesses in a specific area or within a certain distance or drive time from a location.
It enables you to query and use information from a large collection of data sets including population, income, housing,
consumer behavior, and the natural environment.

For example: What kind of people live here? What do people like to do in this area? What are their habits and
lifestyles? What kind of businesses are in this area?

The enrich() method to can be used retrieve demographics and other relevant characteristics associated with the area
surrounding the requested places. You can also use the arcgis.geoenrichment module to obtain additional geographic
context (for example, the ZIP Code of a location) and geographic boundaries (for example, the geometry for a drive-time
service area).

Site analysis is a popular application of this type of data enrichment. For example, GeoEnrichment can be leveraged to
study the population that would be affected by the development of a new community center within their neighborhood.
With the enrich() method, the proposed site can be submitted, and the demographics and other relevant characteristics
associated with the area around the site will be returned.
"""
from .enrichment import *
