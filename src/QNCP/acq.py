import pyvisa
from pyvisa.constants import Parity,StopBits
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from scipy.signal import find_peaks
import inspect
import vxi11
import time
#===========================================================================
# Oscilloscope DS1102Z_E,  Pete (v2.6 tested)
#===========================================================================            
class Rigol_DS1102Z_E:
    def __init__(self,address): # 'TCPIP::<IP ADDRESS>::INSTR'
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.run()

    def meas(self,ch,var):  # measure 
        if type(ch) == str:  # channel info
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
        
        __readOut = self.dev.query(':MEASure:ITEM? {}, CHAN{}'.format(var,__ch))
        __value = float(re.search('.*(?=\n)',__readOut).group())
        return __value
    def run(self):
        self.dev.write(':RUN')
    def stop(self):
        self.dev.write(':STOP')
    def single(self):
        self.dev.write(':SINGle')
    
    def config(self,ch,cfg):  # specifically designed for Pete lock error monitor
        if type(ch) == str:  # channel info, default ch2
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
            
        # pete lock error monitor
        if re.search('(pete)',cfg,re.IGNORECASE) != None:
            __tPID = 120e-6    # PID response time in Pete
            __tDiv = 10e-6    # second/div
            __tOffset = __tPID + 12*__tDiv    # offset
            __vTrig = 0    # trigger level
            __vScale = 0.5    # 500mV/div, 
            self.run()
            self.dev.write(':TIMebase:SCALe {}'.format(__tDiv))   # time scale 10us/div
            self.dev.write(':TIMebase:MAIN:OFFSet {}'.format(__tOffset))            
            self.dev.write(':CHAN{}:COUPling DC'.format(__ch))  # DC coupling
            self.dev.write(':CHAN{}:SCALe {}'.format(__ch,__vScale)) 
            self.dev.write(':CHAN{}:OFFSet 0'.format(__ch))  # DC coupling
            self.dev.write(':TRIG:MODE EDGE')  # trigger on edge
            self.dev.write(':TRIG:SWEep AUTO')  # force trigger when not triggered
            self.dev.write(':TRIG:EDGe:SOURce CHAN{}'.format(__ch))  #trigger
            self.dev.write(':TRIG:EDGe:LEV {}'.format(__vTrig))  
            
#================================================================
# Rigol_DSA832 Spectrum Analyzer ()
#================================================================
class Rigol_DSA832:
    def __init__(self,address): #'TCPIP::<IP ADDRESS>::INSTR' 
        self.address = address            
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.dev.write_termination='\n'
        self.dev.read_termination='\n'

    def freq(self,*f):  # center frequency
        if bool(f) == True:
            __f = f[0]
            self.dev.write(':SENSe:FREQuency:CENTer {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:FREQuency:CENTer?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq

    def span(self,*f):  # frequency range 
        if bool(f) == True: 
            __f = f[0] 
            self.dev.write(':SENSe:FREQuency:SPAN {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:FREQuency:SPAN?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq

    def rbw(self,*f):  # resolution bandwidth
        if bool(f) == True:
            __f = f[0]
            self.dev.write(':SENSe:BANDwidth:RESolution {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:BANDwidth:RESolution?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq
    def vbw(self,*f):  # video bandwidth
        if bool(f) == True:
            __f = f[0]
            self.dev.write(':SENSe:BANDwidth:VIDeo {}'.format(self.__Hz(__f)));
        else:
            __readOut = self.dev.query(':SENSe:BANDwidth:VIDeo?')
            __freq = float(re.search('.*(?=\n)',__readOut).group())*1e-6
            return __freq
        
    def __Hz(self,f):  # in Hz, support unit. Default: MHz
        if type(f) == str:
            if re.search('[mM]',f) != None:
                return 1e6*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[kK]',f) != None:
                return 1e3*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[hH]',f) != None:
                return 1*float(re.sub('[a-zA-Z]','',f))
            elif re.search('[gG]',f) != None:
                return 1e9*float(re.sub('[a-zA-Z]','',f))
        else: # float, or str that contains only numbers
            return float(f)*1e6

    def auto(self): # auto VBW and RBW
        self.dev.write(':SENSe:BANDwidth:RESolution:AUTO ON')
        self.dev.write(':SENSe:BANDwidth:VIDeo:AUTO ON')
        
    def set_frequency_bounds(self,f_lower,f_upper):
        self.dev.write('SENS:FREQ:STARt {}'.format(f_lower))
        self.dev.write('SENS:FREQ:STOP {}'.format(f_upper))        
        
    def monitor(self): ##print spectral signal, can also detect peaks
        
        ##ask for freq limits for plotting
        
        f_lower = float(self.dev.query(':SENS:FREQ:STARt?'))
        f_upper = float(self.dev.query(':SENS:FREQ:STOP?'))
        
        ##initiate
        
        self.dev.write(':INITiate:CONTinuous OFF')
        self.dev.write(':TRACe1:MODE WRITe')
        self.dev.write(':FORMat[:TRACe][:DATA] ASCii')
        
        ##trigger 
        
        self.dev.write(':INITiate')
        while (int(self.dev.query(':STATus:OPERation:CONDition?'))&(1<<3)):
            pass
        
        ##collect and turn into array
        
        rawdata=self.dev.query(':TRACe:DATA? TRACe1',)
        data = rawdata.split(", ")
        data[0] = data[0].split()[1]
        data = np.array([float(i)for i in data])
        freq = np.linspace(f_lower,f_upper,601) #601 is the sample size
        
        ##find peaks and (using peak prominences)
        
        peaks, properties = find_peaks(data, prominence=1,width=20)
        print('Freq. Peaks = ',freq[peaks],'Hz')
        print('Amp. Peaks = ',data[peaks],'dBm')
        
        ##plot
        
        plt.plot(freq,data)
        plt.plot(freq[peaks],data[peaks],'x',ms=15,label=str(freq[peaks]))
        plt.legend()
        plt.xlabel('Freq [Hz]')
        plt.ylabel('Amp [dBm]')
        
        return 
    
    def detect_peaks(self): #find peaks

        ##initiate
        self.dev.write(':INITiate:CONTinuous OFF')
        self.dev.write(':TRACe1:MODE WRITe')
        self.dev.write(':TRACe:MATH:PEAK:TABLe:STATe ON')
        self.dev.write(':FORMat[:TRACe][:DATA] ASCii')
        
        ##trigger 
        self.dev.write(':INITiate')
        while (int(self.dev.query(':STATus:OPERation:CONDition?'))&(1<<3)):
            pass
        
        rawpeaks = self.dev.query(':TRAC:MATH:PEAK?')
        peaks = np.array(rawpeaks.split(",")).astype(np.float)
        freqs=peaks[::2]
        amps=peaks[1::2]
        
        return freqs,amps

#===========================================================================
# Oscilloscope DS1104
#===========================================================================            
class Rigol_DS1104:
    def __init__(self,address):
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.run()

    def meas(self,ch,var):  # measure 
        if type(ch) == str:  # channel info
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
        
        __readOut = self.dev.query(':MEASure:ITEM? {}, CHAN{}'.format(var,__ch))
        __value = float(re.search('.*(?=\n)',__readOut).group())
        return __value
    
    def run(self):
        self.dev.write(':RUN')
    def stop(self):
        self.dev.write(':STOP')
    def single(self):
        self.dev.write(':SINGle')
    
    def screenshot(self,ch):
        self.dev.write(":WAV:SOUR: CHAN{}".format(ch))
        self.dev.write(":WAV:MODE NORMal")
        self.dev.write(":WAV:FORM ASC")
        self.dev.write(":WAV:DATA? CHAN{}".format(ch))
        rawdata = self.dev.read_raw()
        rawdata = rawdata.decode('UTF-8')
        volt_data = rawdata[11:-2] #removes header and ending of data
        volt_data = np.array([float(volt_data) for volt_data in volt_data.split(',')])
        
        t = float(self.dev.query(':WAVeform:XINCrement?'))
        time_data = np.arange(0,t*len(volt_data),t)
        
        return time_data,volt_data

#===========================================================================
# Oscilloscope Rigol DMO5000
#=========================================================================== 

class Rigol_DMO5000:
    def __init__(self,address):
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.address)
        self.run()

    def meas(self,ch,var):  # measure 
        if type(ch) == str:  # channel info
            __ch = re.search('\d',ch).group()
        else:
            __ch = ch
        
        __readOut = self.dev.query(':MEASure:ITEM? {}, CHAN{}'.format(var,__ch))
        __value = float(re.search('.*(?=\n)',__readOut).group())
        return __value
    def run(self):
        self.dev.write(':RUN')
    def stop(self):
        self.dev.write(':STOP')
    def single(self):
        self.dev.write(':SINGle')
        
    def screenshot(self,ch):
        self.dev.write(":WAV:SOUR: CHAN{}".format(ch))
        self.dev.write(":WAV:MODE NORMal")
        self.dev.write(":WAV:FORM ASC")
        self.dev.write(":WAV:DATA? CHAN{}".format(ch))
        rawdata = self.dev.read_raw()
        rawdata = rawdata.decode('UTF-8')
        volt_data = rawdata[11:-2] #removes header and ending of data
        volt_data = np.array([float(volt_data) for volt_data in volt_data.split(',')])
        
        t = float(self.dev.query(':WAVeform:XINCrement?'))
        time_data = np.arange(0,t*len(volt_data),t)
        
        return time_data,volt_data