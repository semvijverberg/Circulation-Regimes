#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:50:16 2018

@author: semvijverberg
"""
import numpy
import random

def ROC_score(predictions, observed, threshold_event):
    
#    predictions = crosscorr_mcK
#    observed = mcKts
#    threshold_event = hotdaythreshold
    
   # calculate ROC scores
    observed = numpy.copy(observed)
    # P_index = numpy.copy(AIR_rain_index)	
    # Test ROC-score			
    
    TP_rate = numpy.ones((11))
    FP_rate =  numpy.ones((11))
    TP_rate[10] = 0
    FP_rate[10] = 0
    AUC_new90 = numpy.zeros((10000))
    
    #print(fixed_event_threshold) 
    events = numpy.where(observed > threshold_event)[0][:]  
    not_events = numpy.where(observed <= threshold_event)[0][:]     
    for p in numpy.linspace(1, 9, 9, dtype=int):	
        
        p_pred = numpy.percentile(predictions, p*10)
        positives_pred = numpy.where(predictions > predictions.mean() + p_pred)[0][:]
        negatives_pred = numpy.where(predictions <= predictions.mean()+ p_pred)[0][:]
        						
        True_pos = [a for a in positives_pred if a in events]
        False_neg = [a for a in negatives_pred if a in events]
        
        False_pos = [a for a in positives_pred if a  in not_events]
        True_neg = [a for a in negatives_pred if a  in not_events]
        
        True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
        False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
        
        FP_rate[p] = False_pos_rate
        TP_rate[p] = True_pos_rate
        
#        threshold =  crosscorr_mcK.mean() + np.percentile(crosscorr_mcK, p*10)
#        pos_prediction_at_lag = np.where( crosscorr_mcK > threshold  )[0]
#        neg_prediction_at_lag = np.where( crosscorr_mcK <= threshold  )[0]
#        dates_min_lag = matchhotdates - pd.Timedelta(int(0), unit='d')
#        true_pos_pred    = [a for a in pos_prediction_at_lag if (a + lag) in hotindex]
#        false_pos_pred   = [a for a in pos_prediction_at_lag if (a + lag) not in hotindex]
#        
#        true_neg_pred    = [a for a in neg_prediction_at_lag if (a + lag) not in hotindex]
#        false_neg_pred    = [a for a in neg_prediction_at_lag if (a + lag) in hotindex]
#        
#        true_pos_rate = len(true_pos_pred) / ( len(true_pos_pred) + len(false_pos_pred) )
#        false_pos_rate = len(false_pos_pred) / ( len(false_pos_pred) + len(true_neg_pred) )
#        	
#        FP_rate[p] = False_pos_rate
#        TP_rate[p] = True_pos_rate
     
    ROC_score = numpy.abs(numpy.trapz(TP_rate, x=FP_rate ))
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
#        fixed_event_threshold = numpy.percentile(P_index, 90)
#        #print(fixed_event_threshold)    
#        events = numpy.where(P_index > fixed_event_threshold)[0][:]  # 10% weakest ISM months
#        not_events = numpy.where(P_index <= fixed_event_threshold)[0][:]         
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
#        AUC_new90[j] = AUC_score
#            

#    print(' * * hot days * * ')
#    print(ROC_score) 
    # calculate p values
#    print('pval')
#    AUC_new90_sort = numpy.sort(AUC_new90[:])
#    pval90  = (numpy.asarray(numpy.where(AUC_new90_sort > ROC_score)).size)/10000.
#    print(pval90)
#    print(' * * * * * * * ')
    return ROC_score