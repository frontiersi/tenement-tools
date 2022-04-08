def convert_ensemble_parameters(in_lyrs):
    """
    Takes list of lists from ArcGIS Pro parameters
    for ensemble modelling tool and converts text
    to numerics.
    """
    
    clean_lyrs = []
    for lyr in in_lyrs:
        a, bc, d = lyr[3], lyr[4], lyr[5]

        # clean a
        if a.lower() == 'na':
            a = None
        elif a.lower() == 'min':
            a == 'Min'
        elif a.lower() == 'max':
            a == 'Max'
        elif a.isnumeric():
            a = float(a)
        else:
            raise ValueError('Value for a is not supported.')

        # clean bc
        if bc.lower() == 'na':
            bc = None
        elif bc.lower() == 'min':
            bc == 'Min'
        elif bc.lower() == 'max':
            bc == 'Max'
        elif bc.isnumeric():
            bc = float(bc)
        else:
            raise ValueError('Value for bc is not supported.')

        # clean d
        if d == '' or d.lower() == 'na':
            d = None
        elif d.lower() == 'min':
            d == 'Min'
        elif d.lower() == 'max':
            d == 'Max'
        elif d.isnumeric():
            d = float(d)
        else:
            raise ValueError('Value for d is not supported.')
            
        # check if two nones in list. max is 1
        num_none = sum(i is None for i in [a, bc, d])
        if num_none > 1:
            raise ValueError('Signoidals do not support two NA values.')
            
        # check if two min in list. max is 1
        num_min = sum(str(i) == 'Min' for i in [a, bc, d])
        if num_min > 1:
            raise ValueError('Signoidals do not support two Min values.')

        # check if two max in list. max is 1
        num_max = sum(str(i) == 'Max' for i in [a, bc, d])
        if num_max > 1:
            raise ValueError('Signoidals do not support two Max values.')                  
        
        # append
        clean_lyrs.append([lyr[0].value, lyr[1], lyr[2], a, bc, d])
            
    # gimme
    return clean_lyrs