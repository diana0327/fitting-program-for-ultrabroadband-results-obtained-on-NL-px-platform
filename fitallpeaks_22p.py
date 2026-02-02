import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from datetime import datetime
import pickle
import os

network_path = r"\\atlas.epfl.ch\lqno\zhiyuan xie\all experiment results\240423 new np cw laser\results for plotting stepping results\forthefirst3peaks\New folder\testwithdifferentphases\1600samplefitting"
os.chdir(network_path)

print("Now in directory:", os.getcwd())
# === Load experimental data ===
path = r'\\atlas.epfl.ch\lqno\zhiyuan xie\all experiment results\240423 new np cw laser\results for plotting stepping results\forthefirst3peaks\New folder\testwithdifferentphases\1600samplefitting\1080doubleclean.xlsx'
df = pd.read_excel(path)
x_data = df['x'].values
SFG_data = df['SFG'].values
DFG_data = df['DFG'].values
intensity_data = df['intensity'].values
SFG_data = SFG_data / np.max(SFG_data)
DFG_data = DFG_data / np.max(DFG_data)
intensity_data = intensity_data / np.max(intensity_data)
# === Fixed frequency and damping ===
wr = [995.3, 1003.2, 1038.61, 1080.7, 1148, 1260, 1277, 1472, 1584, 1596.6]
gamma = [5, 4, 4, 5,5, 6,6, 5, 6, 6]
epsilon = 1e-8

# === Model function ===
def model_function(x, *params):
    ampl = params[0:10]
    phases = params[10:20]
    xnr_phase = params[20]

    x = cp.asarray(x)

    Xr_SFG = sum(a * cp.exp(1j * p) / (w - x - 1j * g / 2 + epsilon)
                 for a, p, w, g in zip(ampl, phases, wr, gamma))
    Xr_DFG = sum(a * cp.exp(1j * p) / (w - x + 1j * g / 2 + epsilon)
                 for a, p, w, g in zip(ampl, phases, wr, gamma))

    Xnr = 1 * cp.exp(1j * xnr_phase)

    X_SFG = Xr_SFG + Xnr
    X_DFG = Xr_DFG + Xnr

    modulus_SFG = cp.abs(X_SFG)**2
    modulus_DFG = cp.abs(X_DFG)**2
    ratio = modulus_SFG / (modulus_DFG + epsilon)

    # Normalize each
    norm = lambda arr: (arr - cp.min(arr)) / (cp.max(arr - cp.min(arr)) + epsilon)
    return norm(modulus_SFG), norm(modulus_DFG), norm(ratio)

# === Cost function ===
def global_cost_function(**kwargs):
    try:
        args = list(kwargs.values())
        mod_SFG, mod_DFG, mod_ratio = model_function(x_data, *args)
        mod_SFG, mod_DFG, mod_ratio = cp.asnumpy(mod_SFG), cp.asnumpy(mod_DFG), cp.asnumpy(mod_ratio)
        cost = (np.sum((SFG_data - mod_SFG)**2) +
                np.sum((DFG_data - mod_DFG)**2) +
                np.sum((intensity_data - mod_ratio)**2)) / 3
        return -cost
    except Exception as e:
        print(f"Error: {e}")
        return 1e6

# === Parameter bounds (only amplitude and phase) ===
pbounds = {
    'Ampl1': (0.0844 , 0.4958),
    'Ampl2': (1.1279 , 1.1587),
    'Ampl3': (1.1245 , 1.2310),
    'Ampl4': (1.1069 , 1.4066),
    'Ampl5': (1.1230 , 1.6567),
    'Ampl6': (2.4632 , 2.8082),
    'Ampl7': (2.0882 , 2.3389),
    'Ampl8': (3.3127 , 3.5549),
    'Ampl9': (1.7807 , 1.8983),
    'Ampl10': (0.3434 , 0.6675),
    'Phase1': (-1.7101 , 1.6621),
    'Phase2': (-2.9394 , -2.4158),
    'Phase3': (-2.7815 , -2.2916),
    'Phase4': (1.1882 , 2.4227),
    'Phase5': (0.8043 , 1.9843),
    'Phase6': (1.5032 , 1.5677),
    'Phase7': (-1.8428 , np.pi),
    'Phase8': (1.1199 , 1.1518),
    'Phase9': (-np.pi , np.pi),
    'Phase10': (-np.pi , np.pi),
    'Xnr_phase': (1.3925 , 1.4367)
}

def save_optimizer(opt, path='optimizer_state.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(opt, f)

def load_optimizer(path='optimizer_state.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            opt = pickle.load(f)
        print(" Loaded previous optimizer state.")
        return opt
    else:
        print(" No previous optimizer state found.")
        return None

# === Run optimization ===
def run_optimization():
    state_path = 'optimizer_state.pkl'
    optimizer = load_optimizer(state_path)

    if optimizer is None:
        optimizer = BayesianOptimization(
            f=global_cost_function,
            pbounds=pbounds,
            verbose=2,
            random_state=1
        )
        init_points = 2000
    else:
        init_points =1  # 已初始化过 

    optimizer.maximize(init_points=init_points, n_iter=2000)
    save_optimizer(optimizer, state_path)
    return optimizer


# === Plot results ===
def plot_results(opt):
    param_order = list(pbounds.keys())
    params = [opt.max['params'][key] for key in param_order]
    mod_SFG, mod_DFG, mod_ratio = model_function(x_data, *params)
    mod_SFG = cp.asnumpy(mod_SFG)
    mod_DFG = cp.asnumpy(mod_DFG)
    mod_ratio = cp.asnumpy(mod_ratio)

    plt.figure(); plt.plot(x_data, SFG_data, label='SFG raw'); plt.plot(x_data, mod_SFG, '--', label='SFG model'); plt.legend(); plt.title('SFG'); plt.grid()
    plt.figure(); plt.plot(x_data, DFG_data, label='DFG raw'); plt.plot(x_data, mod_DFG, '--', label='DFG model'); plt.legend(); plt.title('DFG'); plt.grid()
    plt.figure(); plt.plot(x_data, intensity_data, label='Ratio raw'); plt.plot(x_data, mod_ratio, '--', label='Ratio model'); plt.legend(); plt.title('SFG/DFG'); plt.grid()
    plt.show()

# === Save results ===
def save_results(opt):
    all_results = pd.DataFrame([
    {'target': res['target'], **{k: res['params'].get(k, None) for k in pbounds.keys()}}
    for res in opt.res
    ])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = fr'\\atlas.epfl.ch\lqno\zhiyuan xie\all experiment results\240423 new np cw laser\results for plotting stepping results\forthefirst3peaks\New folder\testwithdifferentphases\1600samplefitting\fitresult_{timestamp}.xlsx'
#    out_path = fr'C:\Users\zhxie\Desktop\temp\fitresult_{timestamp}.xlsx'

    all_results.to_excel(out_path, index=False)
    print(f' All results saved to {out_path}')


# === Execute ===
opt = run_optimization()
plot_results(opt)
save_results(opt)

#pbounds = {
#    'Ampl1': (0.49, 0.55),
#    'Ampl2': (0.35, 0.415),
#    'Ampl3': (0.27, 0.34),
#    'Ampl4': (0.825, 0.84),
#    'Ampl5': (0.68, 0.705),
#    'Ampl6': (1.33, 1.37),
#    'Ampl7': (1.52, 1.58),
#    'Ampl8': (1.17, 1.24),
#    'Phase1': (0.2, 0.26),
#    'Phase2': (-3, -2.93),
#    'Phase3': (-0.72, -0.65),
#    'Phase4': (-0.26, -0.177),
#    'Phase5': (2.8, 2.9),
#    'Phase6': (0.33, 0.37),
#    'Phase7': (-2.9, -2.83),
#    'Phase8': (-0.04, 0.042),
#    'Xnr_phase': (-2.26, -2.235)
#}

