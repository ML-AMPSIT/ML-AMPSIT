#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import os
import threading
import warnings
warnings.filterwarnings('ignore')

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from ampsit.single import create_predict_and_plot_global
from ampsit.utils import start_timer
from ampsit.config import load_config


def start_timer(should_stop, timer_label):
    import time
    start_time = time.time()
    while not should_stop["value"]:
        elapsed = time.time() - start_time
        timer_label.value = f"Timer: {elapsed:.1f}s"
        time.sleep(0.1)

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
methmax=7;
tunmin=0;
tunmax=2;
hourmin=1;
hourmax=totalhours;
Xmin=[vpointmin,hpointmin,varmin,Nmin,methmin,hourmin,tunmin]
Xmax=[vpointmax,hpointmax,varmax,Nmax,methmax,hourmax,tunmax]
Nstep=[verticalmax-1,hpointmax-1,varmax-1, (Nmax - Nmin) // 10, methmax-1, totalhours-1 , tunmax]

def display_custom_css():
    """CSS personalizzato per la GUI"""
    css = """
    <style>
    .ml-ampsit-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #f5f3ee, #e8e2d9)",
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0, 0, 50, 0.1);
        color: #2c3e50;
        margin: 20px 0;
        border: 3px solid #5d6d7e;
    }

    .slider-row {
        background: linear-gradient(135deg, #a1b7cd, #9ab0c5); 
        border: 1px solid #9ab0c5;
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
    }

    .slider-row:hover {
        background: linear-gradient(145deg, #738a9c, #627b8f);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,50,0.1);
    }

    .slider-label {
        color: #2c3e50;
        font-weight: 600;
        font-size: 14px;
        width: 240px;
        text-align: right;
    }

    .methods-legend-custom {
        background: linear-gradient(145deg, #a0b6cb, #94a8bc);
        border: 1px solid #8ca2b8;
        border-radius: 12px;
        padding: 20px;
        color: #2c3e50;
        max-width: 300px;
        margin-left: 30px;
        box-shadow: 0 4px 12px rgba(0, 0, 50, 0.1);
    }

    .main-title-custom {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #2c3e50;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    .subtitle-custom {
        text-align: center;
        font-size: 18px;
        margin-bottom: 15px;
        color: #34495e;
        opacity: 0.85;
    }

    /* Stile slider */
    .widget-slider .ui-slider .ui-slider-handle {
        background-color: #3498db !important;
        border: 2px solid white !important;
        width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3) !important;
    }

    .widget-slider .ui-slider .ui-slider-range {
        background-color: rgba(52, 152, 219, 0.6) !important;
        border-radius: 4px !important;
    }

    .widget-slider .ui-slider {
        background-color: rgba(44, 62, 80, 0.2); !important;
        border-radius: 4px !important;
        height: 8px !important;
    }

    /* Pulsanti */
    .custom-button-style {
        background: linear-gradient(45deg, #3498db, #2980b9) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 30px !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .custom-button-style:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(52, 152, 219, 0.6) !important;
    }

    .stop-button-style {
        background: linear-gradient(45deg, #e74c3c, #c0392b) !important;
        border-radius: 25px !important;
        color: white !important;
        font-weight: bold !important;
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stop-button-style:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(231, 76, 60, 0.6) !important;
    }
    
    a {
    color: #2980b9 !important;
    font-weight: bold;
    text-decoration: underline;
    }
    
    a:hover {
        color: #1c5980 !important;
    }
    
    code {
    background-color: #ecf0f1;
    color: #2c3e50;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
    font-family: 'Courier New', monospace;
    }
    
    </style>
    """
    display(HTML(css))

def interactive_global(_, meth, N, var, hpoint, vpoint, hour, tun, with_button=False):
    
    display_custom_css()
    
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
            layout=widgets.Layout(width='100%', min_width='500px')
        )
        sliders[f] = slider
        slider_widgets.append(slider)

    label_texts = [
        "<div class='slider-label'><b>Vertical level:</b> {}</div>".format(verticalmax),
        "<div class='slider-label'><b>Domain region:</b><br>{}</div>".format(', '.join(regions)),
        "<div class='slider-label'><b>Variable:</b><br>{}</div>".format(', '.join(variables)),
        "<div class='slider-label'><b>Number of simulations:</b><br>Choose sample size</div>",
        "<div class='slider-label'><b>Method:</b><br>1=RF, ..., 7=CART</div>",
        "<div class='slider-label'><b>Timestamp:</b><br>Select hour of interest</div>",
        "<div class='slider-label'><b>Tuning:</b><br>0=Off, 1=On, 2=Load</div>"
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
    stop_button.add_class('stop-button-style')

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
        run_button.add_class('custom-button-style')

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

                should_stop["value"] = True
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

    paired_rows = []
    for label, slider in zip(label_widgets, slider_widgets):
        row = widgets.HBox(
            [label, slider],
            layout=widgets.Layout(
                justify_content='flex-start',
                align_items='center',
                padding='15px',
                margin='8px 0',
            )
        )
        row.add_class('slider-row')
        paired_rows.append(row)

    sliders_and_labels_box = widgets.VBox(
        paired_rows,
        layout=widgets.Layout(
            justify_content='center',
            width='auto',
            padding='10px'
        )
    )

    methods_legend = widgets.HTML(
        """
        <div class="methods-legend-custom">
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

    title = widgets.HTML("""
        <div class='main-title-custom'>ML-AMPSIT</div>
    """)
    subtitle = widgets.HTML("""
        <div class='subtitle-custom'>
            Machine Learning-based Automated Multi-method Parameter Sensitivity and Importance analysis Tool
        </div>
    """)
    subtitle2 = widgets.HTML("""
        <div class='subtitle-custom' style='font-size: 14px;'>
            Interactive UI by <b>Dario Di Santo</b>, Dept. of Civil, Environmental and Mechanical Engineering, University of Trento
        </div>
    """)
    subtitle3 = widgets.HTML("""
        <div class='subtitle-custom' style='font-size: 14px;'>
            Configure the sliders and click RUN. The output will be saved as specified in <code>configAMPSIT.json</code>.<br>
            Project: <a href='https://github.com/ML-AMPSIT/ML-AMPSIT' target='_blank' style='color: #e3f2fd;'>GitHub Repository</a>
        </div>
    """)

    main_container = widgets.VBox([
        title, subtitle, subtitle2, subtitle3,
        widgets.HTML('<hr style="border-color: rgba(255,255,255,0.3);">'),
        legend_and_sliders_box,
        widgets.HTML('<br>'),
        widgets.VBox([
            widgets.HBox([run_button, stop_button], layout=widgets.Layout(justify_content='center', gap='20px')),
            widgets.HBox([timer_label], layout=widgets.Layout(justify_content='center')) 
        ]),
        widgets.HTML('<br>'),
        status_output
    ], layout=widgets.Layout(align_items='center'))
    
    main_container.add_class('ml-ampsit-container')

    return sliders, main_container

