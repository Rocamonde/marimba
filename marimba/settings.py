import os

WORKING_DIR = '/home/rocamonde0_gmail_com/plasticc'

DATA_DIR = os.path.join(WORKING_DIR, 'data')                    # ./data
DATA_FITS_DIR = os.path.join(DATA_DIR, 'fits')                  # ./data/fits
LOGS_DIR = os.path.join(WORKING_DIR, 'logs')                    # ./logs
SAVED_MODELS_DIR = os.path.join(WORKING_DIR, 'models_saved')    # ./models_saved
MODEL_INFO_DIR = os.path.join(WORKING_DIR, 'models_available')  # ./models_available

FIT_CADENCE = 10
FIT_TOTAL_LENGTH = 120
FIT_PAD = 50
FIT_VERSION = f'fit_{FIT_TOTAL_LENGTH}_{FIT_CADENCE}_{FIT_PAD}'

TRAIN_SLICE = 450000
TEST_SLICE = 150000

__all__ = [
    'WORKING_DIR',
    'DATA_DIR',
    'DATA_FITS_DIR',
    'LOGS_DIR',
    'SAVED_MODELS_DIR',
    'MODEL_INFO_DIR',
    'FIT_CADENCE',
    'FIT_TOTAL_LENGTH',
    'FIT_PAD',
    'FIT_VERSION',
    'TRAIN_SLICE',
    'TEST_SLICE',
]
