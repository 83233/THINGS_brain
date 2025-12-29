
rating_cols = [
            'property_manmade_mean','property_precious_mean','property_lives_mean',
            'property_heavy_mean','property_natural_mean','property_moves_mean',
            'property_grasp_mean','property_hold_mean','property_be-moved_mean','property_pleasant_mean'
        ]

SD_cols = [
            'property_manmade_SD','property_precious_SD','property_lives_SD',
            'property_heavy_SD','property_natural_SD','property_moves_SD',
            'property_grasp_SD','property_hold_SD','property_be-moved_SD','property_pleasant_SD'
        ]

def get_rating_cols():
    return rating_cols

def get_SD_cols():
    return SD_cols

def get_all_cols():
    return rating_cols + SD_cols