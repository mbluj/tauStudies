##
## \date August 2022
## \author Michal Bluj
## based on $ROOTSYS/tutorials/tmva/tmva101_Training.py by Stefan Wunsch
##

import ROOT
import numpy as np
import xgboost as xgb
from typing import Tuple
import pickle
import copy
import math
import glob, os
from datetime import datetime
import argparse # it needs to come after ROOT import

from DataPreparation import load_data
import utility_functions as utils

########################
# custom objective funct after https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html
#
# objective func
## Squared error in -pi pi range
# 1st derivative of squared error function with distance (error) in the -pi,pi range convenient for phi angle
def gradient_sqerr_target(predt: np.ndarray, labels: np.ndarray, 
                          target='eta') -> np.ndarray:
    '''Compute the gradient squared error with target dependent definition of "error"'''
    y = labels
    if target=='pt':
        return (predt - y) / y**2
    elif target=='phi':
        df_phi_mpi_pi = np.frompyfunc(utils.phi_mpi_pi,1,1)
        return df_phi_mpi_pi(predt - y)
    else: #if not pt or phi use normal gradient of squared error function (aka 'eta') 
        return (predt - y)
# specialised versions
def gradient_sqerr_pt(predt: np.ndarray, labels: np.ndarray) -> np.ndarray:
    '''Compute the gradient squared percentage (relative) error'''
    return gradient_sqerr_target(predt=predt, labels=labels, target='pt')
def gradient_sqerr(predt: np.ndarray, labels: np.ndarray) -> np.ndarray:
    '''Compute the gradient squared error'''
    return gradient_sqerr_target(predt=predt, labels=labels, target='eta')
def gradient_sqerr_phi(predt: np.ndarray, labels: np.ndarray) -> np.ndarray:
    '''Compute the gradient squared error defined in -pi pi range'''
    return gradient_sqerr_target(predt=predt, labels=labels, target='phi')

########################
# 2nd derivative of squared error function with distance (error) in the -pi,pi range convenient for phi angle
def hessian_sqerr_target(predt: np.ndarray, labels: np.ndarray,
                         target='absolute') -> np.ndarray:
    '''Compute the hessian for squared error with target dependent definition of "error"'''
    y = labels
    if target in ['pt', 'relative']:
        return np.ones(y.shape[0],dtype=float) / y**2
    else: #defelt i.e. absolute, eta, phi or whatever
        return np.ones(y.shape[0],dtype=float)
# specialised versions
def hessian_sqerr(predt: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y = labels
    return hessian_sqerr_target(predt=predt, labels=labels, target='absolute')
def hessian_sqerr_rel(predt: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y = labels
    return hessian_sqerr_target(predt=predt, labels=labels, target='relative')

########################
def squared_err(labels: np.ndarray,
                predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Error objective. A simplified (?) version for RMSE used as
    objective function
    '''
    grad = gradient_sqerr(predt, labels)
    hess = hessian_sqerr(predt, labels)
    return grad, hess
def squared_err_phi(labels: np.ndarray,
                    predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Error objective. A simplified (?) version for RMSE used as
    objective function defined in -pi pi range
    '''
    grad = gradient_sqerr_phi(predt, labels)
    hess = hessian_sqerr(predt, labels)
    return grad, hess
def squared_err_pt(labels: np.ndarray,
                   predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Error objective. A simplified (?) version for RMSPE used as
    objective function (percentage error, i.e. relative to true value)
    '''
    grad = gradient_sqerr_pt(predt, labels)
    hess = hessian_sqerr_rel(predt, labels)
    return grad, hess

########################
def rmse_phi(labels, predt):
    ''' Root mean squared error metric in -pi pi range'''

    #output_errors = np.average((predt - labels)**2, axis=0)
    df_phi_mpi_pi = np.frompyfunc(utils.phi_mpi_pi,1,1)
    output_errors = np.average(df_phi_mpi_pi(predt - labels)**2, axis=0)
    output_errors = np.sqrt(output_errors)

    return output_errors

########################
def rmse_phi_xgb(predt: np.ndarray, y_true: xgb.DMatrix):
    ''' Root mean squared error metric in -pi pi range (XGB interface)'''

    labels = y_true.get_label()
    return 'RMSE_phi', rmse_phi(labels, predt)

########################
def rmse2(labels, predt):
    ''' Root mean squared error metric'''

    output_errors = np.average((predt - labels)**2, axis=0)
    output_errors = np.sqrt(output_errors)

    return output_errors

########################
def rmse2_xgb(predt: np.ndarray, y_true: xgb.DMatrix):
    ''' Root mean squared error metric (XGB interface)'''

    labels = y_true.get_label()
    return 'RMSE2', rmse2(labels, predt)

########################
def rmspe(labels, predt):
    ''' Root mean squared percentage error (i.e. ralative error) metric'''

    output_errors = np.average(((predt - labels) / labels)**2, axis=0)
    output_errors = np.sqrt(output_errors)

    return output_errors

########################
def rmspe_xgb(predt: np.ndarray, y_true: xgb.DMatrix):
    ''' Root mean squared percentage error (i.e. ralative error) metric (XGB interface)'''

    labels = y_true.get_label()
    return 'RMSPE', rmspe(labels, predt)

########################
if __name__ == "__main__":

    # command line arguments parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    utils.addCommonConfArgs(parser)
    args = parser.parse_args()

    target = args.target
    reduced = args.reduced
    sample_label = args.sampleLabel #'signal' / 'tauGun'

    # Load data
    x, y, z = load_data("train_"+sample_label+".root", target=target, reduced=reduced)
    x_v, y_v, z_v = load_data("validate_"+sample_label+".root", target=target, reduced=reduced)

    ##
    # XGBoost model

    # Define model
    #Perhaps the most commonly configured hyperparameters are the following:
    #* n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.
    #* max_depth: The maximum depth of each tree, often values are between 1 and 10.
    #* eta: The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.
    #* subsample: The number of samples (rows) used in each tree, set to a value between 0 and 1, often 1.0 to use all samples.
    #* colsample_bytree: Number of features (columns) used in each tree, set to a value between 0 and 1, often 1.0 to use all features.
    ##
    #n_jobs - #threads    
    objective = None
    eval_metric = None
    if target=='phi':
        objective = squared_err_phi
        eval_metric = rmse_phi_xgb # custom metric
    elif target=='eta':
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
    elif target=='pt': #FIXME
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
        #objective = squared_err_pt
        #eval_metric = rmspe_xgb # custom metric

    bdt = xgb.XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.1,
                           #eval_metric=rmse, # option not available before 1.6.0
                           objective=objective)
                           #objective='reg:squarederror')
                           #objective='reg:linear')#default objctive ('reg:squarederror') unsupported by SaveXGBoost

    print('Model:', bdt, flush=True)

    # Fit final model
    print("Training with", x.shape[0], "events with", x.shape[1], "input features",  flush=True)
    current_time = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
    print("Training started:", current_time, flush=True)
    model_dir = "training/"
    os.makedirs(model_dir, exist_ok=True)
    bdt.fit(x, y,
            eval_set = [(x_v,y_v)],
            eval_metric = eval_metric,
            verbose = False)

    # Save fitted model
    model_name = "model_shift_"
    if reduced:
        model_name += "reduced_"
    model_name += sample_label+"_"+target+"_"+current_time
    print("Saving model in "+model_name+".json", flush=True)
    bdt.save_model(model_dir+model_name+".json")

    # Save model in TMVA format
    print("Saving model with TMVA in "+model_name+".root", flush=True)
    bdt.objective = 'reg:linear' # huck to store model in root format with objective recognised by TMVA
    if hasattr(bdt, "base_score"):
        if bdt.base_score == None:
            bdt.base_score = 0 # set it to trivial value to not mislead TMVA with None
    ROOT.TMVA.Experimental.SaveXGBoost(bdt, "myBDT", model_dir+model_name+".root", num_inputs=x.shape[1])

    current_time = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
    print("Training finished:", current_time, flush=True)
