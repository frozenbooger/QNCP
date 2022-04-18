import inspect
from scipy.ndimage import interpolation
import numpy as np
import matplotlib.pyplot as plt 

class storage_process():
    def __init__(self, total_time):
        """ 
        Description: intializes a storage process which will drive the pulsing 
        on EIT-based quantum memories

        Input: total_time : total time length between repetitions in seconds : float
        Output: class object
        Example: 
        >> sp1 = storage_process(1e-6) 
        
        """
        self.control = []
        self.probe = []
        self.total_time = total_time
        self.buffer_size = 2**14-2
    
    def preparation_crest(self, waveform, *arg):
        """ 
        Description: method that is used by control_process method to construct 
        preparation part of control pulse
        
        Input: waveform : information for preparation control pulse : method, function, or array 
        Output: data : array with voltage information : np.array
        """
        if inspect.ismethod(waveform) == True or inspect.isfunction(waveform) == True:
            
            time = np.linspace(0,self.preparation_time,self.buffer_preparation_time)
            data = waveform(time,*arg)
            
        else:
            
            rawdata = np.array(waveform)
            interp = self.buffer_preparation_time
            z = interp / len(rawdata)
            data = interpolation.zoom(rawdata,z)
            
        return data
    
    def retrieve_crest(self, waveform, *arg):
        """ 
        Description: method that is used by control_process method to construct 
        retrieval part of control pulse
        
        Input: waveform : information for retrieval control pulse : method, function, or array 
        Output: data : array with voltage information : np.array
        """
        
        if inspect.ismethod(waveform) == True or inspect.isfunction(waveform) == True:
            
            time = np.linspace(0,self.retrieval_time,self.buffer_retrieval_time)
            data = waveform(time,*arg)
            
        else:
            
            rawdata = np.array(waveform)
            interp = self.buffer_retrieval_time
            z = interp / len(rawdata)
            data = interpolation.zoom(rawdata,z)
            
        return data
        
    def control_process(self, storage_time: float, preparation_time: float, preparation_process, retrieval_time: float, retrieval_process):
        """ 
        Description: This method serves as the constructor for the control pulse
        
        Input: storage_time : storage time : float
               preparation_time : preparation pulse time : float
               preparation_process : method/function with args or list : list
               retrieval_time :retrieval pulse time : float
               retrieval_process : method/function with args or list : list
               
        Example: 
        >> sp1.control_process(1e-6, 2e-7, [[2]], 3e-7, [[4]])
        """
        self.storage_time = storage_time
        self.preparation_time = preparation_time
        self.retrieval_time = retrieval_time
        
        if self.storage_time + self.preparation_time + self.retrieval_time > self.total_time:
            raise ValueError('timing parameters are infeasable')
            
        self.extra_time = self.total_time - (self.storage_time + self.preparation_time + self.retrieval_time)
        buffer_list = np.linspace(0,self.total_time,self.buffer_size)
        
        self.buffer_wait_time_1 = int(np.around(((self.extra_time/2)/self.total_time)*self.buffer_size,0))
        self.buffer_preparation_time = int(np.around(((self.preparation_time)/self.total_time)*self.buffer_size,0))
        self.buffer_retrieval_time = int(np.around(((self.retrieval_time)/self.total_time)*self.buffer_size,0))
        self.buffer_strorage_time = int(np.around(((self.storage_time)/self.total_time)*self.buffer_size,0))
        self.buffer_wait_time_2 = self.buffer_size-(self.buffer_wait_time_1+self.buffer_preparation_time+self.buffer_retrieval_time+self.buffer_strorage_time)
        
        preparation_array = self.preparation_crest(*preparation_process)
        retrieval_array = self.retrieve_crest(*retrieval_process)
        storage_array = [0]*self.buffer_strorage_time
        wait_time_1 = [0]*self.buffer_wait_time_1
        wait_time_2 = [0]*self.buffer_wait_time_2 
        
        self.control_array = [*wait_time_1,*preparation_array,*storage_array,*retrieval_array,*wait_time_2]
    
    @staticmethod
    def gaussian(t,mu,FWHM,a): #Gaussian Function. Inputs: (Center, FWHM, Amplitude)
        sigma = (FWHM)/(2*np.sqrt(2*np.log(2)))
        amplitude = np.sqrt(2*np.pi*sigma**2)*a
        return amplitude*( 1/(sigma * np.sqrt(2*np.pi) ) )*np.exp( -((t-mu)**2 / (2*sigma**2)) )
        
    def probe_process(self, FWHM, amplitude, *offset):
        """ 
        Description: This method serves as the constructor for the probe pulse
        
        Input: FWHM : FWHM : float
               amplitude : amplitude : float
               offset : offset : float
               
        Example: 
        >> sp1.probe_process(1e-6, 2, -1e-7)
        """
        self.time_array = np.linspace(0,self.total_time,self.buffer_size)
        if bool(offset) == True:
            center = self.time_array[self.buffer_wait_time_1+self.buffer_preparation_time]+offset
        else:
            center = self.time_array[self.buffer_wait_time_1+self.buffer_preparation_time]
            
        self.probe_array =  self.gaussian(self.time_array, center, FWHM, amplitude)
        
    def plots(self):
        """ 
        Description: Plots the probe and control functions

        Input: None: 
        Output: Plot
        Example: 
        >> sp1.plots()
        """
        plt.plot(self.time_array, self.control_array, label='CONTROL')
        plt.plot(self.time_array, self.probe_array, label='PROBE')
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [s]')
        plt.show()       