#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:17:05 2020

@author: apatane
"""

def my_relu(arr):
    arr = arr * (arr > 0)
    return arr