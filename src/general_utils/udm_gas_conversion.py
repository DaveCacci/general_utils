import logging

def udm_gas_conversion(data, T, Pt, V, udm: str):
    '''
    udm is the original unit of the data
    '''
    if udm == 'mmolLd':
        data = data*(0.083145*T/Pt)/1000/24*V
    elif udm == 'Lh':
        data = data/(0.083145*T/Pt)*1000*24/V
    else:
        logging.info('UDM not recognized, must be either "mmolLd" or "Lh"')
    return data