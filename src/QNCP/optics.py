import pyvisa
import time
from ctypes import *
import numpy as np
import time

class thorlabs_paddles:
    def __init__(self,path):
        """ 
        Description: The initialization of the thorlabs_paddles class 
        will initialize a virtual controller which can be used to control
        multiple paddles

        Input: Thorlabs Kinesis Path on Computer : string
        Output: device object with all its methods : class object
        Example: 
        >>thorlabs_paddles('C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.Polarizer.dll')

        thorlabs_paddles.class.object...
        """
        self.lib = cdll.LoadLibrary(path) 
        self.devices=[]
        self.serials=[]

        if self.lib.TLI_BuildDeviceList() != 0:
            raise Exception("Could not build device list (KinesisLib __init__)")
        self.lib.MPC_GetPosition.restype=c_double
        self.lib.MPC_GetStatusBits.restype=c_uint
            

    def add_device(self,serial):
        """ 
        Description: Adds paddle system to be controlled, an index is 
        assigned to each device by order of when it was added relative 
        to paddles already with an assigned index

        Input: Serial code of device : string
        Output: add_device Method : class method
        Example: 
        >>add_device('38137934')
        """
        serial_no = c_char_p(bytes(serial, "utf-8"))
        self.serials.append(serial)
        self.devices.append(serial_no)

        if self.lib.MPC_Open(serial_no) != 0:
            raise Exception("Could not open device with serial number "+ serial)

        time.sleep(1.0)
        self.lib.MPC_StartPolling(serial_no, c_int(200))
        self.lib.MPC_ClearMessageQueue(serial_no)
        time.sleep(1.0)

    def move_paddle(self,device_id,paddle_id,position):
        """ 
        Description: Moves the desired paddle.

        Input: device index : int, paddle index : int, absolute angle (0-170) : float
        Output: Moves paddles : class method
        Example: 
        >>move_paddle(1,2,170) # moves the second paddle of the first device to 170 degrees
        """
        self.lib.MPC_MoveToPosition(self.devices[device_id], c_int(paddle_id), c_double(position))

    def stop_paddle(self,device_id,paddle_id):
        """ 
        Description: Stops the desired paddle.

        Input: device index : int, paddle index : int
        Output: Stokes Parameters : Tuple
        Example: 
        >>stop_paddle(1,2) # stops the second paddle of the first device
        """
        self.lib.MPC_Stop(self.devices[device_id],c_int(paddle_id))

    def get_paddle_position(self,device_id,paddle_id):
        """ 
        Description: Returns the current paddle position of the desired device

        Input: device index : int, paddle index : int
        Output: Positions of paddles : List
        Example: 
        >>get_paddle_position(1,2) 

        [20, 34, 130]
        """
        return self.lib.MPC_GetPosition(self.devices[device_id],c_int(paddle_id))   
    
class thorlabs_waveplates:
    def __init__(self,path):
        """ 
        Description: The initialization of the thorlabs_waveplates class 
        will initialize a virtual controller which can be used to control
        kinesis cubes

        Input: Thorlabs Kinesis Path on Computer : string
        Output: device object with all its methods : class object
        Example: 
        >>thorlabs_waveplates('C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.Polarizer.dll')

        thorlabs_waveplates.class.object...
        """
        self.lib = cdll.LoadLibrary(path) 
        self.devices=[]
        self.serials=[]

        if self.lib.TLI_BuildDeviceList() != 0:
            raise Exception("Could not build device list")
            

    def add_device(self, serial):
        """ 
        Description: Adds kinesis cube systems to be controlled, an index is 
        assigned to each device by order of when it was added relative 
        to kinesis cubes already with an assigned index

        Input: Serial code of device : string
        Output: addDevice Method : class method
        Example: 
        >>add_device('38137934')
        """
        serial_no = c_char_p(bytes(serial, "utf-8"))
        self.serials.append(serial)
        self.devices.append(serial_no)

        if self.lib.CC_Open(serial_no) != 0:
            raise Exception("Could not open device with serial number "+ serial)

        time.sleep(1.0)
        self.lib.CC_StartPolling(serial_no, c_int(200))
        self.lib.CC_Home(serial_no)
        time.sleep(1.0)

    def move_motor(self,device_id,position):
        """ 
        Description: moves motor of device to desired position

        Input: device index : int, motor index : int
        Output: None : none
        Example: 
        >>move_motor(1,2) 
        """
        self.TL.CC_ClearMessageQueue(self.devices[device_id])
        self.TL.CC_MoveToPosition(self.devices[device_id], c_double(position))    
    
    def stop_motor(self,device_id):
        """ 
        Description: Stops the desired motor.

        Input: device index : int, paddle index : int
        Output: Stokes Parameters : Tuple
        Example: 
        >>stop_motor(1,2) # stops the second paddle of the first device
        """
        
    def get_motor_position(self,device_id):
        """
        Return current position in abslute device units
        """ 
        return self.TL.CC_GetPosition(self.devices[device_id])
