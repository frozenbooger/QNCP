import redpitaya_scpi as scpi

class red_pitaya:
    def __init__(self,address):
        self.address = address
        self.rp_s = scpi.scpi(self.address)
        
    def set_pin(self, pin : str, direction : bool, *state : bool):
        """
        Description: Sets the direction and state of pin
        
        Inputs: pin : see pin diagram : str
                direction : in or out : bool
                state : optional if out : bool
        
        Example: >>> set_pin('7_P', 1, 1) # turns pin DIO7_P on
        """
        directions = ['IN','OUT']
        if direction == 1:
            direction_command = 'DIG:PIN:DIR {}'.format(directions[1]) + ',DIO' + pin
            self.rp_s.tx_txt(direction_command)
            state_command = 'DIG:PIN DIO' + pin + ',{}'.format(state[0]) 
            self.rp_s.tx_txt(state_command)
        if direction == 0:
            direction_command = 'DIG:PIN:DIR {}'.format(directions[0]) + ',DIO' + pin
            self.rp_s.tx_txt(direction_command)
            output_state = self.rp_s.txrx_txt('DIG:PIN? ' + 'DIO' + pin)
            return int(output_state)