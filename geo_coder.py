from opencage.geocoder import OpenCageGeocode


def get_neighborhood_name(lat, lon, gdf=None, spatial_index=None):

    lat = str(lat) + '01'
    lon = str(lon) + '01'
    dict1 = get_geographic_info_opencage(lat, lon)
    return dict1["neighborhood"] if dict1["neighborhood"] else dict1["road"]

from googletrans import Translator

def translate_hebrew_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='he', dest='en')
    return translation.text
