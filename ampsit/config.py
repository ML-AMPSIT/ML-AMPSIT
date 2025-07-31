#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
#def load_config(path='E:/secondPAPER/newGRASS/VARStxtFOR_TEST_PURPOSES/configAMPSITvalleynewGRASS.json'):
def load_config(path='../configAMPSIT.json'):
    with open(path) as f:
        return json.load(f)

#def load_loop_config(path='E:/secondPAPER/newGRASS/VARStxtFOR_TEST_PURPOSES/loopconfig.json'):
def load_loop_config(path='../loopconfig.json'):
    with open(path, 'r') as f:
        return json.load(f)

