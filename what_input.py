#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:51:50 2018

@author: semvijverberg
"""

class Variable:
    import os
    from datetime import datetime, timedelta
    base_path = '/Users/semvijverberg/surfdrive/Data_ERAint/'
    path_input = os.path.join(base_path, 'input_pp')
    path_raw = os.path.join(base_path, 'input_raw')
    if os.path.isdir(path_input):
        pass
    else:
        print("{}\n\npath input does not exist".format(path_input))
    def __init__(self, name, dataset, startyear, endyear, startmonth, endmonth, grid, tfreq):
        # self is the instance of the employee class
        # below are listed the instance variables
        self.name = name
        self.startyear = startyear
        self.endyear = endyear
        self.startmonth = startmonth
        self.endmonth = endmonth
        self.grid = grid
        self.tfreq = tfreq
        self.dataset = dataset
        filename = '{}_{}-{}_{}_{}_dt-{}_{}'.format(self.name, self.startyear, self.endyear, self.startmonth, self.endmonth, self.tfreq, self.grid).replace(' ', '_').replace('/','x')
        self.filename = filename +'.nc'
        print("Variable function selected {} \n".format(self.filename))
