#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:50:16 2018

@author: semvijverberg
"""
import numpy
import random

def ROC_score(predictor, observed, threshold_event):
    
    predictor = crosscorr_mcK
    observed = mcKtsfull
    threshold_event = mcKts.mean(dim='time').values + mcKts.std().values
    
   # calculate ROC scores
    observed = numpy.copy(observed)
    # P_index = numpy.copy(AIR_rain_index)	
    # Test ROC-score			
    
    TP_rate = numpy.ones((11))
    FP_rate =  numpy.ones((11))
    TP_rate[10] = 0
    FP_rate[10] = 0
    AUC_new90 = numpy.zeros((10000))
    
    fixed_event_threshold = numpy.percentile(P_index, 90)
    #print(fixed_event_threshold) 
    events = numpy.where(observed > fixed_event_threshold)[0][:]  # 10% strongest ISM months
    not_events = numpy.where(observed <= fixed_event_threshold)[0][:]     
    for p in numpy.linspace(1, 9, 9):	
        

        positives_pred = numpy.where(predictor > threshold_event)[0][:]
        negatives_pred = numpy.where(predictor <= threshold_event)[0][:]
        						
        True_pos = [a for a in positives_pred if a in events]
        False_neg = [a for a in negatives_pred if a in events]
        
        False_pos = [a for a in positives_pred if a  in not_events]
        True_neg = [a for a in negatives_pred if a  in not_events]
        True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
        False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
        
        FP_rate[p] = False_pos_rate
        TP_rate[p] = True_pos_rate
        	

     
    ROC_score90 = numpy.abs(numpy.trapz(TP_rate, FP_rate))

    # shuffled ROc
    
    
#    for j in range(10000):
#        
#        old_index = range(0,len(predictor),1)
#                
#        sample_index = random.sample(old_index, len(old_index))
#        #print(sample_index)
#        new_obs_rain = predictor[sample_index]    
#        # _____________________________________________________________________________
#        # calculate new AUC score and store it
#        # _____________________________________________________________________________
#        #
#    
#        new_index = new_obs_rain
#        P_index = numpy.copy(new_index)
#        # P_index = numpy.copy(MT_rain_index)	
#        # Test AUC-score			
#        TP_rate = numpy.ones((11))
#        FP_rate =  numpy.ones((11))
#        TP_rate[10] = 0
#        FP_rate[10] = 0
#        
#        fixed_event_threshold = numpy.percentile(P_index, 10)
#        #print(fixed_event_threshold)    
#        events = numpy.where(P_index < fixed_event_threshold)[0][:]  # 10% weakest ISM months
#        not_events = numpy.where(P_index >= fixed_event_threshold)[0][:]         
#        for p in numpy.linspace(1, 9, 9):	
#            
#               
#            #p_pred = numpy.percentile(Y_pred_validation, p*10)
#            p_pred = numpy.percentile(Y_pred_validation, p*10)
#            positives_pred = numpy.where(Y_pred_validation > p_pred)[0][:]
#            negatives_pred = numpy.where(Y_pred_validation <= p_pred)[0][:]
#            						
#            True_pos = [a for a in positives_pred if a in events]
#            False_neg = [a for a in negatives_pred if a in events]
#            
#            False_pos = [a for a in positives_pred if a  in not_events]
#            True_neg = [a for a in negatives_pred if a  in not_events]
#            
#            #check
#            if len(True_pos+False_neg) != len(events) :
#                print("check 136")
#            elif len(True_neg+False_pos) != len(not_events) :
#                print("check 138")
#           
#            True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
#            False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
#            
#            FP_rate[p] = False_pos_rate
#            TP_rate[p] = True_pos_rate
#        
#        AUC_score = numpy.abs(numpy.trapz(TP_rate, FP_rate))
#        AUC_new10[j, i] = AUC_score
    
    # 70th 
    
    
    for j in range(10000):
        
        old_index = range(0,len(predictor),1)
                
        sample_index = random.sample(old_index, len(old_index))
        #print(sample_index)
        new_obs_rain = predictor[sample_index]    
        # _____________________________________________________________________________
        # calculate new AUC score and store it
        # _____________________________________________________________________________
        #
    
        new_index = new_obs_rain
        P_index = numpy.copy(new_index)
        # P_index = numpy.copy(MT_rain_index)	
        # Test AUC-score			
        TP_rate = numpy.ones((11))
        FP_rate =  numpy.ones((11))
        TP_rate[10] = 0
        FP_rate[10] = 0
        
        fixed_event_threshold = numpy.percentile(P_index, 90)
        #print(fixed_event_threshold)    
        events = numpy.where(P_index > fixed_event_threshold)[0][:]  # 10% weakest ISM months
        not_events = numpy.where(P_index <= fixed_event_threshold)[0][:]         
        for p in numpy.linspace(1, 9, 9):	
            
               
            #p_pred = numpy.percentile(Y_pred_validation, p*10)
            p_pred = numpy.percentile(Y_pred_validation, p*10)
            positives_pred = numpy.where(Y_pred_validation > p_pred)[0][:]
            negatives_pred = numpy.where(Y_pred_validation <= p_pred)[0][:]
            						
            True_pos = [a for a in positives_pred if a in events]
            False_neg = [a for a in negatives_pred if a in events]
            
            False_pos = [a for a in positives_pred if a  in not_events]
            True_neg = [a for a in negatives_pred if a  in not_events]
            
            #check
            if len(True_pos+False_neg) != len(events) :
                print("check 136")
            elif len(True_neg+False_pos) != len(not_events) :
                print("check 138")
           
            True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
            False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
            
            FP_rate[p] = False_pos_rate
            TP_rate[p] = True_pos_rate
        
        AUC_score = numpy.abs(numpy.trapz(TP_rate, FP_rate))
        AUC_new90[j] = AUC_score
            

    print(' * * 90 * * ')
    print(ROC_score90) 
    # calculate p values
    print('pval')
    AUC_new90_sort = numpy.sort(AUC_new90[:])
    pval90  = (numpy.asarray(numpy.where(AUC_new90_sort > ROC_score90)).size)/10000.
    print(pval90)
    print(' * * * * * * * ')