import numpy as np
import logging

def convert_u_to_pwm(u, u_max, dt: int, pump_dose_per_minute = 25, period: int = 15):
    '''
    Convert the control signal 'u' to the PWM signal for the pumps
    u: float, control signal
    u_max: float, maximum control signal
    dt: int, control interval in seconds
    pump_dose_per_minute: float, dose of the pump in mL/min
    period: int, lenght of the PWM signal in seconds
    Note: this works for 'NMPC', but not for 'Selector PI', since the controller outputs a delta u value (nominal shall be added)
    '''
    try:
        pump_dose_per_period = pump_dose_per_minute * period/60 #mL/period_sec; 4L/h = 66.67mL/min = 1.11mL/s; 1.5L/h = 25mL/min = 0.42mL/s

        required_volume = u/(24*60*60)*dt #mL
        
        max_on_periods = np.round(u_max/(24*60*60)*dt/pump_dose_per_period,0) # Only for precaution/safety purposes as a double-check

        on_periods_tot = max(0, min(np.round(required_volume / pump_dose_per_period, 0), max_on_periods)) #how many periods of period_sec on
        
        periods_in_dt = dt/period #Amount of period_sec periods within the control time interval ('dt' in seconds)
        if on_periods_tot > 0: 
            dosing_interval = np.round(periods_in_dt/on_periods_tot, 0) #Devide control interval in equal 'on_minutes_tot' fractions
            on_periods_ini = 1 #Turn on the pump for one minute/control interval fraction
            off_periods_ini = dosing_interval - 1
        else:
            dosing_interval = periods_in_dt #If 'saturationLow', don't turn on pumps for any minute in the control interval
            on_periods_ini = 0
            off_periods_ini = dosing_interval

        #Compute values to be placed in .ini file (range 0...86400 seconds)
        on_seconds_ini = max(0, min(on_periods_ini*period,86400)) #sec
        off_seconds_ini = max(0, min(off_periods_ini*period,86400)) #sec
        conversion_error = 'No'
    except Exception as e:
        conversion_error = e
        logging.error(e)
    return (on_seconds_ini, off_seconds_ini), on_periods_tot, conversion_error