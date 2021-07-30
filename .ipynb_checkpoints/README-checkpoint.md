# Using device 

## Introduction

The purpose of device is to streamline and versitalize your laboratory workflow. This document will guide you through the prerequisites and use of this library.

## Prerequisites

This library will require you download the packages in labinstruments-packages.txt. To do so use the following command:

    pip install labinstruments-packages.txt
    
or if conda is used:

    conda install labinstruments-packages.txt
    
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

    gen.Rigol_DSG830.freq(50)
    
In general:

    type.instrument.command(input)