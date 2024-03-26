# Copyright (C) Jan 2020 Mellanox Technologies Ltd. All rights reserved.   
#                                                                           
# This software is available to you under a choice of one of two            
# licenses.  You may choose to be licensed under the terms of the GNU       
# General Public License (GPL) Version 2, available from the file           
# COPYING in the main directory of this source tree, or the                 
# OpenIB.org BSD license below:                                             
#                                                                           
#     Redistribution and use in source and binary forms, with or            
#     without modification, are permitted provided that the following       
#     conditions are met:                                                   
#                                                                           
#      - Redistributions of source code must retain the above               
#        copyright notice, this list of conditions and the following        
#        disclaimer.                                                        
#                                                                           
#      - Redistributions in binary form must reproduce the above            
#        copyright notice, this list of conditions and the following        
#        disclaimer in the documentation and/or other materials             
#        provided with the distribution.                                    
#                                                                           
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,         
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF        
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                     
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS       
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN        
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN         
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE          
# SOFTWARE.                                                                 
# --                                                                        


#######################################################
# 
# ResDumpCommand.py
# Python implementation of the Class ResDumpCommand
# Generated by Enterprise Architect
# Created on:      14-Aug-2019 10:11:58 AM
# Original author: talve
# 
#######################################################
from abc import ABC, abstractmethod


class ResDumpCommand(ABC):
    """This class is an interface for the resource dump commands and perform a
    strategy pattern of the execution by calling the internal methods.
      validate, get_data and print_data that will be implemented by each command.
    """
    def execute(self):
        """Command execution call:
           1. validate.
           2. get_data.
           3. print_data.
        """
        self.validate()
        self.get_data()
        self.print_data()

    @abstractmethod
    def get_data(self):
        """get the needed data.
        """
        pass

    @abstractmethod
    def print_data(self, data):
        """Print the collected data.
        """
        pass

    @abstractmethod
    def validate(self):
        """validate.
        """
        pass
