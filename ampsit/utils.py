#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def nearest_divisible(n , lst):
  size = len(lst)
  remainder = n % (size+2)
  if remainder == 0:
    return n
  diff = size - remainder
  return n + diff

import matplotlib.pyplot as plt

def get_distinct_colors(n):
    base_colors = list(plt.get_cmap('tab20').colors)  # 20 colori distinti
    extra_needed = n - len(base_colors)

    if extra_needed <= 0:
        return base_colors[:n]
    
    # Genera colori supplementari (es. con 'hls' per buona separazione)
    import seaborn as sns
    extra_colors = sns.color_palette("hls", extra_needed)

    return base_colors + extra_colors
  
import time
def start_timer(stop_flag, label):
    start = time.time()
    while not stop_flag["value"]:
        elapsed = time.time() - start
        label.value = f"Elapsed time: {elapsed:.1f}s"
        time.sleep(1)
        
        
