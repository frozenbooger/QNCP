# Using QNCP 

## Introduction

QNCP is currently a library of drivers necessary to the control and monitoring of experimental laboratory setups needing analog signals. A homogenous language of commands has been created towards devloping hardware agnostic protocols for experiments. These drivers lay the bedrock which will serve to configure laboratory instruments to perform parallel and in-series protocols needed in quantum communication experiments. The library has been written over python in order to make the infrastructure accessible and easy to use. 

## Installation

To install simply type in 

    pip install QNCP
    
To update to newest version
    
    pip install --upgrade QNCP
    
## Instruments 

The different instruments supported by this library are divided into the following categories:

* Generator Instruments (gen)
    
* Acquisition Instruments (acq)
    
For example calling a function generator from the Rigol DG4000 family to change its output frequency to 50 Mhz is as follows:

	from QNCP import gen

    gen.Rigol_DG4000.freq(50)
    
This library also has a search functionality for USB (ASRL, GBIP) devices. We can search as follows:

	from QNCP import search, device

	print(search.get_resource('ASRL',device.Quantum_Composers,38400))
