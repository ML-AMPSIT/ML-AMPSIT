#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import os
import threading
import warnings
warnings.filterwarnings('ignore')

import ipywidgets as widgets
from IPython.display import display, clear_output

from ampsit.single import create_predict_and_plot_global
from ampsit.utils import start_timer

import json
from ampsit.config import load_config
config = load_config()

totalhours = config['totalhours']
variables = config['variables']
regions = config['regions']
verticalmax = config['verticalmax']
totalsim = config['totalsim']
parameter_names = config['parameter_names']
output_path = config['output_pathname']
tun_iter = config['tun_iter']

from ampsit.utils import nearest_divisible
n_sample_calc = nearest_divisible(1000, parameter_names)

choices=['vpoint','hpoint','variable','N_simul','method','hour','tuning']
vpointmin=1;
vpointmax=verticalmax;
hpointmin=1;
hpointmax=len(regions);
varmin=1;
varmax=len(variables);
Nmin=totalsim*1/10;
Nmax=totalsim;
methmin=1;
methmax=7; ###
tunmin=0;
tunmax=2;
hourmin=1;
hourmax=totalhours;
Xmin=[vpointmin,hpointmin,varmin,Nmin,methmin,hourmin,tunmin]
Xmax=[vpointmax,hpointmax,varmax,Nmax,methmax,hourmax,tunmax]
Nstep=[verticalmax-1,hpointmax-1,varmax-1, (Nmax - Nmin) // 10, methmax-1, totalhours-1 , tunmax]

def interactive_global(_, meth, N, var, hpoint, vpoint, hour, tun, with_button=False):
    Xnew = [vpoint, hpoint, var, N, meth, hour, tun]
    
    sliders = {}
    slider_widgets = []
    label_widgets = []

    for i, f in enumerate(choices):
        SliderClass = widgets.IntSlider if f in ['vpoint', 'hpoint', 'variable', 'N_simul', 'method', 'hour', 'tuning'] else widgets.FloatSlider
        slider = SliderClass(
            value=int(Xnew[i]) if SliderClass == widgets.IntSlider else Xnew[i],
            min=int(Xmin[i]) if SliderClass == widgets.IntSlider else Xmin[i],
            max=int(Xmax[i]) if SliderClass == widgets.IntSlider else Xmax[i],
            step=1 if SliderClass == widgets.IntSlider else (Xmax[i] - Xmin[i]) / Nstep[i],
            layout=widgets.Layout(width='400px', margin='0 0 0 40px')
        )
        sliders[f] = slider
        slider_widgets.append(slider)

    label_texts = [
        "<div style='width:240px; text-align:right;'><b>Vertical levels:</b> {}</div>".format(verticalmax),
        "<div style='width:240px; text-align:right;'><b>Domain horizontal point:</b><br>{}</div>".format(', '.join(regions)),
        "<div style='width:240px; text-align:right;'><b>Variable:</b><br>{}</div>".format(', '.join(variables)),
        "<div style='width:240px; text-align:right;'><b>Number of simulations:</b><br>Choose sample size</div>",
        "<div style='width:240px; text-align:right;'><b>Methods:</b><br>1=RF, ..., 7=CART</div>",
        "<div style='width:240px; text-align:right;'><b>Hour:</b><br>Select hour of interest</div>",
        "<div style='width:240px; text-align:right;'><b>Tuning:</b><br>0=Off, 1=On, 2=Load</div>"
    ]

    label_widgets = [widgets.HTML(txt) for txt in label_texts]

    status_output = widgets.Output()
    timer_label = widgets.Label(value="Timer: 0.0s", layout=widgets.Layout(margin='10px 0'))

    should_stop = {"value": False}

    stop_button = widgets.Button(
        description="STOP",
        layout=widgets.Layout(width='100px', height='45px'),
        tooltip="Interrupt the process",
        button_style='danger'
    )
    stop_button.style.font_weight = 'bold'
    stop_button.style.font_size = '16px'

    def stop_execution(_):
        should_stop["value"] = True
        status_output.clear_output()
        with status_output:
            display(widgets.HTML("<b style='color: red;'>Execution interrupted by user.</b>"))

    stop_button.on_click(stop_execution)


    run_button = None
    if with_button:
        run_button = widgets.Button(
            description="RUN",
            layout=widgets.Layout(width='180px', height='45px'),
            tooltip="Run the model with selected configuration"
        )
        run_button.style.button_color = '#0078D7'
        run_button.style.font_weight = 'bold'
        run_button.style.font_size = '16px'





        def update_results_within(_):
            should_stop["value"] = False
            status_output.clear_output()
            timer_label.value = "Timer: 0.0s"

            with status_output:
                display(widgets.HTML("<div style='color: green; font-weight: bold;'>Running model... Please wait.</div>"))

            vpt     = sliders['vpoint'].value
            hpt     = sliders['hpoint'].value
            var_val = sliders['variable'].value
            N_val   = sliders['N_simul'].value
            mth     = sliders['method'].value
            hr      = sliders['hour'].value
            tun_val = sliders['tuning'].value

            if tun_val == 1:
                with status_output:
                    display(widgets.HTML("<i style='color: orange;'>Tuning is on — it may require a while…</i>"))

            # Avvia il thread per il timer
            timer_thread = threading.Thread(target=start_timer, args=(should_stop, timer_label))
            timer_thread.start()

            try:
                if should_stop["value"]:
                    raise RuntimeError("Process was interrupted by user.")

                on_change_global = create_predict_and_plot_global(
                    mth, N_val, var_val, hpt, vpt, hr, tun_val, n_sample_calc
                )

                if should_stop["value"]:
                    raise RuntimeError("Process was interrupted by user.")

                on_change_global()

                if should_stop["value"]:
                    raise RuntimeError("Process was interrupted by user.")

                should_stop["value"] = True  # ferma il timer
                timer_thread.join()

                with status_output:
                    status_output.clear_output()
                    display(widgets.HTML(
                        f"<div style='color: blue; font-weight: bold;'>Output generated successfully.</div>"
                    ))

            except Exception as e:
                should_stop["value"] = True
                timer_thread.join()
                with status_output:
                    status_output.clear_output()
                    display(widgets.HTML(f"<b style='color: red;'>Error: {str(e)}</b>"))



        run_button.on_click(update_results_within)

    # Pair labels with sliders
    paired_rows = []
    for label, slider in zip(label_widgets, slider_widgets):
        row = widgets.HBox(
            [label, slider],
            layout=widgets.Layout(
                justify_content='flex-start',
                align_items='center',
                padding='10px',
                border='1px solid #ddd',
                margin='6px 0',
                border_radius='6px',
                background_color='#f9f9f9',
                gap='60px'
            )
        )
        paired_rows.append(row)

    sliders_and_labels_box = widgets.VBox(
        paired_rows,
        layout=widgets.Layout(
            justify_content='center',
            width='auto',
            padding='10px'
        )
    )

    # === Methods Legend ===
    methods_legend = widgets.HTML(
        """
        <div style="text-align: left; padding: 12px 20px; border: 1px solid #ccc; border-radius: 8px; background-color: #f4f4f4; font-size: 14px; line-height: 1.6; max-width: 280px;">
        <b>Available algorithms</b><br>
        (1) Random Forest<br>
        (2) Lasso<br>
        (3) Support Vector Regression<br>
        (4) Bayesian regression<br>
        (5) Gaussian regression<br>
        (6) XGBoost<br>
        (7) CART
        </div>
        """
    )

    legend_and_sliders_box = widgets.HBox(
        [sliders_and_labels_box, methods_legend],
        layout=widgets.Layout(
            justify_content='center',
            align_items='flex-start',
            gap='40px',
            width='100%'
        )
    )

    # === Headers ===
    title = widgets.HTML("""
        <div style='text-align:center; font-family:"Segoe UI", sans-serif; font-size: 38px; font-weight: bold; color: #2c3e50; margin-top: 15px;'>
            ML-AMPSIT
        </div>
    """)
    subtitle = widgets.HTML("""
        <div style='text-align:center; font-family:Arial; font-size: 18px; color: #555;'>
            Machine Learning-based Automated Multi-method Parameter Sensitivity and Importance analysis Tool
        </div>
    """)
    subtitle2 = widgets.HTML("""
        <div style='text-align:center; font-size: 14px; color: #888;'>
            Interactive UI by <b>Dario Di Santo</b>, Dept. of Civil, Environmental and Mechanical Engineering, University of Trento
        </div>
    """)
    subtitle3 = widgets.HTML("""
        <div style='text-align:center; font-size: 14px; color: #666;'>
            Configure the sliders and click RUN. The output will be saved as specified in <code>configAMPSIT.json</code>.<br>
            Project: <a href='https://github.com/ML-AMPSIT/ML-AMPSIT' target='_blank'>GitHub Repository</a>
        </div>
    """)

    # === Final UI ===
    ui = widgets.VBox([
        title, subtitle, subtitle2, subtitle3,
        widgets.HTML('<hr>'),
        legend_and_sliders_box,
        widgets.HTML('<br>'),
        widgets.VBox([
            widgets.HBox([run_button, stop_button], layout=widgets.Layout(justify_content='center', gap='20px')),
            widgets.HBox([timer_label], layout=widgets.Layout(justify_content='center')) 
        ]),
        widgets.HTML('<br>'),
        status_output
    ], layout=widgets.Layout(align_items='center'))

    return sliders, ui



