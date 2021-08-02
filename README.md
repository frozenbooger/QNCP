# Using device 

## Introduction

The purpose of device is to streamline and versitalize your laboratory workflow. This document will guide you through the prerequisites and use of this library.

## Installation

To install simply type in 

    pip install QNCP
    
## Instruments 

The different instruments supported by this library are divided into the following categories:

* Generators (gen)
    * Valon 5105
    * Rigol DSG830
    * Rigol DG4202
    * Quantum Composer
    * Agilent ESG Signal Generators
    
* Analyzers (acq)
    * Rigol DS1102Z E
    * Rigol DSA832
    
* Miscellaneous (misc)
    * MogLabs Agile RF Synthesizer
    
For example calling the Rigol DSG830 to change its output frequency to 50 Mhz is as follows:

	from QNCP import device

    device.Rigol_DSG830.freq(50)
    
This library also has a search functionality for USB (ASRL, GBIP) devices. We can search as follows:

	from QNCP import search, device

	print(search.get_resource('ASRL',device.Quantum_Composers,38400))