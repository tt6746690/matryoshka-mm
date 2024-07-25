import importlib
import copy
from .model_config import *


## automatically add additional model_config

ModelConfig = {}

module = importlib.import_module('llava.config.model_config')
dict_names = [x for x in dir(module) if 
              isinstance(getattr(module, x), dict) and \
              x.startswith('model_config') and \
              x not in ['model_config_pretune', 'model_config_finetune']
              ]
for dict_name in dict_names:
    if any(x in dict_name for x in ['pretune', 'fintune']):
        raise ValueError('should not contain pretune/finetune since will make one for each.')
    for finetune_type in ['pretune', 'finetune']:
        d = copy.deepcopy(getattr(module, dict_name))
        if finetune_type == 'pretune':
            d['use_alternative'] = d.get('use_alternative', True)
        elif finetune_type == 'finetune':
            d['use_alternative'] = d.get('use_alternative', True)
        ModelConfig.update({finetune_type + '_' + dict_name.split('model_config_')[1]: d})
