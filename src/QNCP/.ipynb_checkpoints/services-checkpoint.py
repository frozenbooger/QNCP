import pyvisa
import time
import numpy as np
import time
from qncp import acq, gen, search
import matplotlib.pyplot as plt
import statistics
import random
from tqdm.notebook import tnrange
from numpy.random import rand
from scipy.optimize import basinhopping
from scipy.optimize import curve_fit
import re
import os
import requests 
import warnings

warnings.filterwarnings('ignore')

class AM_characterization:
    def __init__(self, generator, oscilloscope, gench, acqch):
        """ 
        Description: Initializes the charcterization of any amplitude 
        modulation crystal. Requires an oscilloscope and function generator 
        channels. 

        Input: generator : drives amplitude modualtion : class object
               oscilloscope : monitors signal : class object
               gench : generator channel : int
               acqch : acquistion channel : int
        Output: 

        Example: 
        >>AM_Characterization(lithium, oxygen, 1, 3)
        
        """
        self.gen = generator
        self.acq = oscilloscope
        self.gench = gench
        self.acqch = acqch
        self.output_list = []
        self.output_std_list = []
        self.signal_width = 1e-6
        self.waveform = self.gen.square
        self.starting_point = 0
        self.width = self.signal_width*0.5
        self.acq.channel_state(self.acqch,1)
        self.stoch_bool = 0
        self.acq.dev.write(':TIMebase:MAIN:SCALe {}'.format(1/2*self.signal_width))
        
    @staticmethod
    def AM_curve(x,Pmin,Pmax,V0,Vpi):
        """ 
        Description: This describes the curve which models the behavior
        of an amplitude modulator under a changing input voltage
        Input: x : input voltage : float or float array
               Pmin : minimum output power read, fitted : float 
               Pmax : maximum output power read, fitted : float 
               V0 : input voltage which gives maximum : float
               Vpi : range of input voltage which provides full mod. depth: float
        Output: voltage curve : float or float array

        Example: 
        >>AM_curve(4, *params)

        3.5
        """
        return Pmin + (Pmax-Pmin) * (0.5 * np.cos(np.pi*(x-V0)/Vpi) + 0.5)
    
    def initial_calibration(self):
        """ 
        Description: Calibrates the scale of the oscilloscope given the 
        current reading. Several runs will provide optimal results. Used 
        prior to charcterization step

        Input: None : None
        Output: None : None

        Example: 
        >>initial_calibration()

        """
        samples = 5
        
        # Define Bounds
        self.DC_bounds = [-5,5]
        self.step_size = 0.1
        self.input_list = np.arange(self.DC_bounds[0],self.DC_bounds[1]+self.step_size,self.step_size)

        # Run Scaling calibration
        
        volts_list = []
        volts_list_std = []
        state = 1 
        self.acq.channel_state(self.acqch,state)
        measurments = 30
        offset = self.DC_bounds[0]
        self.gen.DC(self.gench,offset)
        self.gen.on()

        for j in tnrange(measurments, desc='Calibrating Measurement'):
            times, volts = self.acq.screenshot(self.acqch)
            volts_list.append(volts)
            volts_list_std.append(statistics.stdev(volts))
        volts_list = np.array(volts_list)
        volts_list_std = np.array(volts_list_std)
        volts_avg = np.average(volts_list, axis=0)
        volt_0 = sum(volts_avg)/len(volts_avg)
        volt_0_std = (1/(len(volts_list_std)))*np.sqrt(sum(volts_list_std**2))

        self.scale  = volt_0/4
        self.offs = -3.5*self.scale
        self.acq.scale_offset(self.acqch,self.scale,self.offs)
        
    def characterize(self):
        """ 
        Description: Performs a scan of the modulator output for different 
        input voltages. Crucial to finding optimal parameters

        Input: None
        Output: None

        Example: 
        >>characterize()
        
        """
        
        self.initial_calibration()
        self.initial_calibration()
        
        samples = 5

        # Run Data Collection Algorithm

        for i in tnrange(len(self.input_list), desc = 'Characterizing EOM'):

            self.gen.DC(self.gench,self.DC_bounds[0]+i*self.step_size)
            volts_list = []
            volts_list_std = []

            for j in range(samples):

                times, volts = self.acq.screenshot(self.acqch)
                volts_list.append(volts)
                volts_list_std.append(statistics.stdev(volts))

            volts_avg = np.average(np.array(volts_list), axis=0)
            volts_std = np.array(volts_list_std)
            volts_avg_std = np.sqrt(sum(volts_std**2))/len(volts_std)
            self.output_std_list.append(volts_avg_std)
            volt_mean = sum(volts_avg)/len(volts_avg)
            self.output_list.append(volt_mean)

            if self.scale < np.abs(volt_mean)/4 :
                self.scale  = np.abs(volt_mean)/4
                self.offs = -3.5*self.scale
                self.acq.scale_offset(self.acqch,self.scale,self.offs)
    
    def estimate_max_min(self):
        """ 
        Description: Performs a fit on the experimental data using 
        AM_curve() to find the optimal parameters

        Input: None : None
        Output: None : None

        Example: 
        >>estimate_max_min()

        """
        if len(self.output_list) == 0:
            self.characterize()
        else:
            pass
        
        min_power = min(self.output_list)
        max_power = max(self.output_list)
        bounds_guess=([-np.inf,-np.inf,-10,0],[np.inf,np.inf,10,10] )
        guess = [min_power,max_power,self.input_list[self.output_list.index(max_power)],np.abs(self.input_list[self.output_list.index(max_power)]-self.input_list[self.output_list.index(min_power)])]
        self.params,self.cov = curve_fit(self.AM_curve,self.input_list,self.output_list,p0=guess,bounds=bounds_guess)
        
        # Determine the maximum and minimum given the limitations of the generator
        if self.params[2]+self.params[3] > self.DC_bounds[-1]:
            self.params[3] = -self.params[3] 
            
        self.v_min = self.params[2]+self.params[3]
        self.v_max = self.params[2]
        
        # Find local minima an maxima if results are out of bounds
        fitted_curve = self.AM_curve(self.input_list,*self.params)
        if self.v_min < 5 or self.v_min > 5:
            local_minima = min(fitted_curve)
            self.v_min = self.input_list[np.where(fitted_curve == local_minima)[0]][0]
        if self.v_max < 5 or self.v_max > 5:
            local_maxima = max(fitted_curve)
            self.v_max = self.input_list[np.where(fitted_curve == local_maxima)[0]][0]
        if self.stoch_bool == 1:
            self.v_max = self.stoch_v_max
            self.v_min = self.stoch_v_min
        
    def plot_characterization(self):
        """ 
        Description: Plots fitting and reports fitted parameters

        Input: None : None
        Output: None : None

        Example: 
        >>plot_characterization()

        """
        if bool(self.output_list) == 0:
            self.characterize
        else:
            pass
        
        self.estimate_max_min()
        
        fig, ax = plt.subplots()
        plt.errorbar(self.input_list, self.output_list, yerr = self.output_std_list, label='raw data', alpha = 0.5)
        plt.plot(self.input_list,self.AM_curve(self.input_list,*self.params),label='fit')
        plt.vlines(self.params[3]+self.params[2],self.params[0],self.params[1])
        plt.title('Fitting to Sinosoid')

        textstr = '\n'.join((r'$P_{min}+(P_{max}-P_{min})\left(\frac{1}{2}\cos\left(\frac{\pi(V-V_0)}{V_\pi}\right)+\frac{1}{2}\right)$','P_min: {}'.format(round(self.params[0],4)),'P_max: {}'.format(round(self.params[1],4)),'V_0 {}'.format(round(self.params[2],4)),'V_pi: {}'.format(round(self.params[3],4))))
        props = dict(facecolor='white', alpha=0.2)
        ax.text(0.03, 0.92, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        plt.style.use('dark_background')
        plt.rcParams["figure.figsize"] = (10,5)
        plt.legend()
        plt.show()

    @staticmethod
    def square(t,leadingedge,width,lower,upper):
        """ 
        Description: Used to perform temporal fitting of the square pulse
        needed to extract the paramters to do stochastic search

        Input: t : time array : array
               leadingedge : leading edge of pulse : float
               width : width of the pulse : float
               lower : lower bound (volatge) of pulse : float
               upper : upper bound (volatge) of pulse : float
        Output: square pulse : fitted curve

        Example: 
        >>square(t, *params)
        
        [0,0,...,0,0,0,1,1,1,...,1,1,1,0,0,...,0,0]
        """
        return np.piecewise(t,[(t<=leadingedge),((t>leadingedge) & (t<leadingedge+width)),(t>=leadingedge+width)],[lower,upper,lower])

    def send_pulses_gen(self, maximum, minimum):
        """ 
        Description: Sends pulses given the minimum maximum specification

        Input: maximum : input voltage which gives maximum output : float
               minimum : input voltage which gives minimum output : float
        Output: None : sends pulses : None

        Example: 
        >>send_pulses_gen_1(1,0)

        """
        if maximum - minimum < 0:
            self.gen.arbitrary(self.gench,self.signal_width, self.waveform(np.linspace(0,self.signal_width,1000),self.starting_point,self.width,-1))
            self.gen.offset(self.gench,minimum-0.5*np.abs(maximum - minimum))
        else:
            self.gen.arbitrary(self.gench,self.signal_width, self.waveform(np.linspace(0,self.signal_width,1000),self.starting_point,self.width,1))
            self.gen.offset(self.gench,minimum+0.5*np.abs(maximum - minimum))
        self.gen.lev(self.gench,np.abs(maximum - minimum))
        mode = 0
        cycles = 1
        self.gen.burst(self.gench, mode, cycles)
        time.sleep(1)
        
        
    def get_max_min_acq(self):
        """ 
        Description: Find the maximum and minimum of pusle read on 
        oscilloscope

        Input: None : None
        Output: maximum of acquired pulses : float
                minimum of acquired pulses : float

        Example: 
        >>get_max_min_acq()
        
        """
        samples = 5
        volts_list = []
        for i in range(samples):
            times, volts = self.acq.screenshot(self.acqch)
            volts_list.append(volts)
        volts_avg = np.average(np.array(volts_list),axis=0)
        diff_volts = np.diff(volts_avg)
        time1 = times[np.where(diff_volts == max(diff_volts))]
        time2 = times[np.where(diff_volts == min(diff_volts))]
        params, cov = curve_fit(self.square, times, volts_avg, p0 = [time1[0],time2[0]-time1[0],min(volts),max(volts)])
        return params[3], params[2]

    def visibility(self, x):
        """ 
        Description: Finds the visibility of a square pulse

        Input: x : maximum and minimum inputs : 1x2 array 
        Output: visibility : float

        Example: 
        >>visibility([-1,4])

        """        
        maximum, minimum = x
        self.acq.scale_offset(self.acqch, self.scale, self.offs)
        self.send_pulses_gen(maximum, minimum)
        max_v, min_v = self.get_max_min_acq()
        return -(max_v-min_v)/(max_v+min_v)

    def stochastic_search(self):
        """ 
        Description: Run stochastic search to find optimal parameters.
        Time consuming task however is robust and mostly agrees with 
        curve and parameters found

        Input: None : None
        Output: None : optimized parameters : None

        Example: 
        >>stochastic_search()
        
        """
        self.estimate_max_min()
        output_max = self.params[1]
        output_min = self.params[0]
        scale = (output_max-output_min)*(1/6)
        offs = -3.5*scale
        self.acq.scale_offset(self.acqch, scale, offs)
        self.acq.trigger_set(self.acqch, 0, np.round((output_max-output_min)/2,4))
        self.gen.reset()
        self.send_pulses_gen(self.v_max,self.v_min)
        self.gen.on()
        # Starting Point
        guess = np.array([self.v_max,self.v_min])
        # Run Basinhopping
        result = basinhopping(self.visibility, guess, niter = 20, niter_success = 2)
        # Result Stats
        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])
        # Result
        solution = result['x']
        evaluation = objective_visibility(solution)
        print('Visibility is {} if [max,min] = {}'.format(-evaluation, solution))
        self.stoch_v_max = solution[0]
        self.stoch_v_min = solution[1]
        self.stoch_bool = 1
        
    def send_gaussian(self, signal_width, center, fwhm):
        """ 
        Description: Sends optimized gaussian pulses to oscilliscope

        Input: signal_width : width of signal : float
               center : peak of gaussian : float
               fwhm : fwhm : float
        Output: None : None

        Example: 
        >>send_gaussian(1e-6,0.5e-6,0.2e-6)
        
        """
        self.estimate_max_min()
        output_max = self.AM_curve(self.v_max,*self.params)
        output_min = self.AM_curve(self.v_min,*self.params)
        scale = (output_max-output_min)*(1/6)
        offs = -3.5*scale
        self.acq.scale_offset(self.acqch, scale, offs)
        self.acq.trigger_set(self.acqch, 0, np.round((output_max-output_min)/2,4))
        waveform = self.gen.gaussian
        maximum = self.v_max
        minimum = self.v_min
        if maximum - minimum < 0:
            self.gen.arbitrary(self.gench,signal_width,waveform,center,fwhm,-1)
            self.gen.offset(self.gench,minimum-0.5*np.abs(maximum - minimum))
        else:
            self.gen.arbitrary(self.gench,signal_width,waveform,center,fwhm,1)
            self.gen.offset(self.gench,minimum+0.5*np.abs(maximum - minimum))
        self.gen.lev(self.gench,np.abs(maximum - minimum))
        mode = 0
        cycles = 1
        self.gen.burst(self.gench, mode, cycles)
        self.gen.on()
        
    def send_max_DC(self):
        """ 
        Description: Sends maximum DC

        Input: None : None
        Output: None : None

        Example: 
        >>send_max_DC()

        """
        self.estimate_max_min()
        self.gen.DC(self.gench, self.v_max)

class qrng:
    def __init__(self,address):
        """ 
        Description: initializes session with QRNG given IP address

        Input: ip address : string
        Output: None
        Example: 
        >> qrng('10.1.2.80')
        
        """
        self.address = 'https://' + address
        
    def get_numbers(self, num_type_set, minimum, maximum, quantity):
        """ 
        Description: gets random numbers of different types

        Input: num_type_set : int = {0='int',1='short',2='float',3='double'}
        Output: List of random numbers : list of floats
        Example: 
        >> qrng1.get_numbers(1,0,5,20) #returns 10 integers between 0 and 100
        
        """
        num_types = ['int','short','float','double']
        num_type = num_types[num_type_set]
        command_num = '/api/2.0/{}?min={}&max={}&quantity={}'.format(num_type,minimum,maximum,quantity) 
        response = requests.get(self.address+command_num,verify=False)
        code = response.text
        refined_code = code.strip('][').split(',')
        refined_code = list(map(float,refined_code))
        return refined_code
    
    def get_bytes(self, size, *quantity):
        """ 
        Description: Still in working on this

        Input: Still in working on this
        Output: Still in working on this
        Example: Still in working on this
        >> 
        
        """
        try:
            if bool(quantity) == True:
                command_bytes = '/api/2.0/hexbytes?DataLength={}&quantity={}'.format(size,quantity) 
            else:
                command_bytes = '/api/2.0/streambytes?size={}'.format(size)  
            response = requests.get(self.address+command_bytes,verify=False)
            code = response.text
            refined_code = code.strip('][').split(',')
            refined_code = list(map(float,refined_code))
            return refined_code
        except:
            raise ValueError('Function is under construction')
    
    def get_performance(self):
        """ 
        Description: gets performance in terms of qrn generation rate

        Input: None
        Output: None
        Example: get_performance()
        >> get_performance()
        
        """
        command_performance = '/api/2.0/performance'
        response = requests.get(self.address+command_performance,verify=False)
        code = response.text
        return code+'Mbps'
    
    def get_info(self):
        """ 
        Description: gets software and firmware editions and info

        Input: None
        Output: None
        Example: get_info()
        >> get_info()
        
        """
        command_firmware = '/api/2.0/firmwareinfo'
        response_firmware = requests.get(self.address+command_firmware,verify=False)
        command_software = '/api/2.0/softwareinfo'
        response_software = requests.get(self.address+command_software,verify=False)
        firmware = response_firmware.text
        software = response_software.text
        return firmware+'\n'+software