# ensemble
"""
"""

# import required libraries
import os, sys
import numpy as np
import pandas as pd
import xarray as xr

#sys.path.append('../../shared')
#import satfetcher, tools

# meta
def bpa_all(ds_dict):
    """
    like, pheno, sdm, chm and frax
    """
    
    # belief (SITE)
    # like * pheno
    m_1 = (ds_dict['like'] * ds_dict['pheno']) + (
          (1 - ds_dict['pheno']) * ds_dict['like']) + (
          (1 - ds_dict['like']) * ds_dict['pheno'])

    # sdm * chm
    m_2 = (ds_dict['sdm'] * ds_dict['chm']) + (
          (1 - ds_dict['chm']) * ds_dict['sdm']) + (
          (1 - ds_dict['sdm']) * ds_dict['chm'])

    # m_1 * m_2
    m_3 = (m_1 * m_2) + (
          (1 - m_2) * m_1) + (
          (1 - m_1) * m_2)

    # disbelief (NONSITE)
    m_4 = ds_dict['frax']
    
    # generate final belief (site) layer
    da_belief = (m_3 * (1 - m_4)) / (1 - (m_4 * m_3))

    # generate final disbelief (non-site) layer
    da_disbelief = (m_4 * (1 - m_3)) / (1 - (m_4 * m_3))

    # generate plausability layer
    da_plauability = (1 - da_disbelief)
    
    # combine into dataset
    ds = xr.merge([
        da_belief.to_dataset(name='belief'), 
        da_disbelief.to_dataset(name='disbelief'), 
        da_plauability.to_dataset(name='plausability')
    ])
    
    return ds


# meta
def bpa_lscv(ds_dict):
    """
    like, sdm, chm and frax only
    """
    
    # belief (SITE)
    m_1 = (ds_dict['like'] * ds_dict['sdm']) + (
          (1 - ds_dict['sdm']) * ds_dict['like']) + (
          (1 - ds_dict['like']) * ds_dict['sdm'])

    # sdm * chm
    m_2 = (m_1 * ds_dict['chm']) + (
          (1 - ds_dict['chm']) * m_1) + (
          (1 - m_1) * ds_dict['chm'])

    # disbelief (NONSITE)
    m_3 = ds_dict['frax']
    
    # generate final belief (site) layer
    da_belief = (m_2 * (1 - m_3)) / (1 - (m_3 * m_2))

    # generate final disbelief (non-site) layer
    da_disbelief = (m_3 * (1 - m_2)) / (1 - (m_3 * m_2))

    # generate plausability layer
    da_plauability = (1 - da_disbelief)
    
    # combine into dataset
    ds = xr.merge([
        da_belief.to_dataset(name='belief'), 
        da_disbelief.to_dataset(name='disbelief'), 
        da_plauability.to_dataset(name='plausability')
    ])
    
    return ds


# meta
def bpa_pscv(ds_dict):
    """
    pheno, sdm, chm and frax only
    """
    
    # belief (SITE)
    m_1 = (ds_dict['pheno'] * ds_dict['sdm']) + (
          (1 - ds_dict['sdm']) * ds_dict['pheno']) + (
          (1 - ds_dict['pheno']) * ds_dict['sdm'])

    # sdm * chm
    m_2 = (m_1 * ds_dict['chm']) + (
          (1 - ds_dict['chm']) * m_1) + (
          (1 - m_1) * ds_dict['chm'])

    # disbelief (NONSITE)
    m_3 = ds_dict['frax']
    
    # generate final belief (site) layer
    da_belief = (m_2 * (1 - m_3)) / (1 - (m_3 * m_2))

    # generate final disbelief (non-site) layer
    da_disbelief = (m_3 * (1 - m_2)) / (1 - (m_3 * m_2))

    # generate plausability layer
    da_plauability = (1 - da_disbelief)
    
    # combine into dataset
    ds = xr.merge([
        da_belief.to_dataset(name='belief'), 
        da_disbelief.to_dataset(name='disbelief'), 
        da_plauability.to_dataset(name='plausability')
    ])
    
    return ds


# meta
def bpa_psv(ds_dict):
    """
    pheno, sdm, frax only
    """
    
    # belief (SITE)
    m_1 = (ds_dict['pheno'] * ds_dict['sdm']) + (
          (1 - ds_dict['sdm']) * ds_dict['pheno']) + (
          (1 - ds_dict['pheno']) * ds_dict['sdm'])

    # disbelief (NONSITE)
    m_2 = ds_dict['frax']
    
    # generate final belief (site) layer
    da_belief = (m_1 * (1 - m_2)) / (1 - (m_2 * m_1))

    # generate final disbelief (non-site) layer
    da_disbelief = (m_2 * (1 - m_1)) / (1 - (m_2 * m_1))

    # generate plausability layer
    da_plauability = (1 - da_disbelief)
    
    # combine into dataset
    ds = xr.merge([
        da_belief.to_dataset(name='belief'), 
        da_disbelief.to_dataset(name='disbelief'), 
        da_plauability.to_dataset(name='plausability')
    ])
    
    return ds


# meta
def bpa_pcv(ds_dict):
    """
    pheno, chm, frax only
    """
    
    # belief (SITE)
    m_1 = (ds_dict['pheno'] * ds_dict['chm']) + (
          (1 - ds_dict['chm']) * ds_dict['pheno']) + (
          (1 - ds_dict['pheno']) * ds_dict['chm'])

    # disbelief (NONSITE)
    m_2 = ds_dict['frax']
    
    # generate final belief (site) layer
    da_belief = (m_1 * (1 - m_2)) / (1 - (m_2 * m_1))

    # generate final disbelief (non-site) layer
    da_disbelief = (m_2 * (1 - m_1)) / (1 - (m_2 * m_1))

    # generate plausability layer
    da_plauability = (1 - da_disbelief)
    
    # combine into dataset
    ds = xr.merge([
        da_belief.to_dataset(name='belief'), 
        da_disbelief.to_dataset(name='disbelief'), 
        da_plauability.to_dataset(name='plausability')
    ])
    
    return ds


# meta
def bpa_pv(ds_dict):
    """
    pheno and frax only
    """
    
    # belief (SITE)
    m_1 = ds_dict['pheno']

    # disbelief (NONSITE)
    m_2 = ds_dict['frax']
    
    # generate final belief (site) layer
    da_belief = (m_1 * (1 - m_2)) / (1 - (m_2 * m_1))

    # generate final disbelief (non-site) layer
    da_disbelief = (m_2 * (1 - m_1)) / (1 - (m_2 * m_1))

    # generate plausability layer
    da_plauability = (1 - da_disbelief)
    
    # combine into dataset
    ds = xr.merge([
        da_belief.to_dataset(name='belief'), 
        da_disbelief.to_dataset(name='disbelief'), 
        da_plauability.to_dataset(name='plausability')
    ])
    
    return ds


