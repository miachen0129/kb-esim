import os


FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

MAX_ENTITIES = 1

# Special token words.
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
ENT_TOKEN = '[ENT]'
SUB_TOKEN = '[SUB]'
PRE_TOKEN = '[PRE]'
OBJ_TOKEN = '[OBJ]'

NEVER_SPLIT_TAG = [
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN,
    ENT_TOKEN, SUB_TOKEN, PRE_TOKEN, OBJ_TOKEN
]
