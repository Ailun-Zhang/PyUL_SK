#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:14:05 2021
Yourong F Wang (FWPhys.github.io)

PyUltraLight 2 performance benchmark.

Usage:
    python Benchmark.py                      # asks for the resolution on a terminal
    python Benchmark.py --resol 64 --yes     # non-interactive (batch jobs, CI)

30.50 (SK maintenance release): command-line options, no input() when the
session is not interactive, no automatic 'pip install' (it tried to install the
standard-library module 'multiprocessing'), outputs written to <repo>/Bench/,
and the momentum/COM-correcting soliton now uses its intended mass MF.
"""

import argparse
import os
import sys

PRESETS = {'potato': 64, 'a': 256, 'b': 384, 'c': 512, 'd': 1024}

parser = argparse.ArgumentParser(description="PyUltraLight 2 performance benchmark")
parser.add_argument("--resol", type=int, default=None,
                    help="grid resolution (presets: Potato=64, A=256, B=384, C=512, D=1024)")
parser.add_argument("--yes", action="store_true", help="do not wait for ENTER")
parser.add_argument("--axion-mass", default="1e-21", help="PYUL_AXION_MASS_EV (default 1e-21)")
parser.add_argument("--out", default=None, help="output folder (default: <repository>/Bench)")
args = parser.parse_args()

interactive = sys.stdin is not None and sys.stdin.isatty()

print("Welcome to PyUltraLight 2 Performance Benchmark")

if args.resol is not None:
    resol = int(args.resol)
elif interactive:
    print("Please key in the letter for a specific resolution. ENTER to confirm")
    print("[Potato] 64^3. [A] 256^3. [B] 384^3. [C] 512^3. [D] 1024^3")
    try:
        userinput = input()
    except EOFError:
        userinput = ''
    resol = PRESETS.get(userinput.strip().lower(), 512)
else:
    resol = 64
    print("Non-interactive session and no --resol given: using 64^3.")

HERE = os.path.dirname(os.path.abspath(__file__))
save_path = args.out or os.path.join(HERE, 'Bench')

print(f"Note: This script will produce outputs in the path {save_path}.")

print("\n")

import numpy as np

if interactive and not args.yes:
    print("Please press ENTER again to proceed.")
    try:
        input()
    except EOFError:
        pass

os.environ["PYUL_AXION_MASS_EV"] = str(args.axion_mass) # Set axion mass you need (read at import)
sys.path.insert(0, HERE)
try:
    import PyUltraLight2 as PyUL
except ImportError as err:
    raise SystemExit(
        f"Could not import PyUltraLight2 ({err}).\n"
        "Install the requirements first, for example\n"
        "    pip install -r requirements.txt            (core)\n"
        "    pip install -r requirements-full.txt       (with pyFFTW, h5py, numba)\n"
        "or, from the repository root, pip install -e .[all]")


###### Do not touch
MinVersion = 23

if (PyUL.S_version < MinVersion):
    raise RuntimeError("You need the latest PyUL!")


np.set_printoptions(suppress=True)

import platform
import time

# Useful Aux Functions
ToCode = PyUL.convert
ToPhys = PyUL.convert_back
CB = PyUL.convert_between
printU = PyUL.printU

# Simulation Parameters

if not os.path.exists(save_path):
    os.makedirs(save_path)
    

length, length_units = 16, ''  #
duration, duration_units = 0.5, ''  #

start_time = 0.  # For solitons only: Pre-evolve the wavefunction phase.

NS = 36

save_format = 'npy'  # npy, npz, hdf5

step_factor = 1

save_number = -1

PyUL.DispN(duration, duration_units, length, length_units, resol, step_factor,
           save_number)

Save_Options = 'Energy NBody DF'

save_options = PyUL.SaveOptionsDigest(Save_Options)

s_mass_unit = ''
s_position_unit = ''
s_velocity_unit = ''

m_mass_unit = ''
m_position_unit = ''
m_velocity_unit = ''

NSols = 8
MassMin = 9
MassMax = 15

GenRange = length/3

particles = []
embeds = []
solitons = []

VMax = 1

Momentum = np.zeros(3)
COM = np.zeros(3)

np.random.seed(seed = 100)

for i in range(NSols):
    
    Mass = np.random.random()*(MassMax-MassMin) + MassMin
    
    Position = np.random.random(3)*GenRange - GenRange/2
    
    Velocity = np.random.random(3)*VMax - VMax / 2
    
    Momentum += Mass * Velocity
    
    COM += Mass * Position
    
    solitons.append([Mass,Position.tolist(),Velocity.tolist(),np.random.random()*2*np.pi])
 
MF = 15

CorrP = -COM/MF
CorrV = -Momentum/MF

# 30.50: the correcting soliton has mass MF (it used the last random Mass, so
# the total momentum and COM were not actually zeroed).
solitons.append([MF,CorrP.tolist(),CorrV.tolist(),0])

COM += MF*CorrP

Momentum += MF*CorrV

gridspace = PyUL.MeshSpacing(resol,length,length_units, silent = True)
# PLUMMER RADIUS (IN LENGTH UNITS)
rP = gridspace / 2

a = PyUL.GenPlummer(rP,length_units)

NBo = 12
for i in range(NBo):
    
    Mass = (np.random.random()*(MassMax-MassMin) + MassMin)*0.05
    
    Position = np.random.random(3)*GenRange - GenRange/2
    
    Velocity = np.random.random(3)*VMax - VMax / 2
    
    particles.append([Mass,Position.tolist(),Velocity.tolist()])
 

printU(f"Initial condition is set-up using {NSols+1} solitons and {NBo} particles.",'Init')


Uniform = False
UVel = [0,0,0]
Density, density_unit = 0, ''

Name = ''

run_folder = PyUL.GenerateConfig(NS, length, length_units,
                                 resol, duration, duration_units, step_factor,
                                 save_number, Save_Options, save_path,
                                 save_format, s_mass_unit, s_position_unit,
                                 s_velocity_unit, solitons,start_time,
                                 m_mass_unit, m_position_unit, m_velocity_unit,
                                 particles,embeds, Uniform,Density,density_unit,
                                 a,UVel,True,Name)


Start_Time = time.time()

PyUL.evolve(save_path,
            run_folder, 
            EdgeClear = False,      # Reflexive Boundary Condition (Half Baked)
            DumpInit = False,       # Dump Initial Wavefunction
            DumpFinal = False,      # Dump Final Wavefunction
            UseInit = False,        # Use Initial Wavefunction (Specify in InitPath)
            IsoP = False,           # Zero Padded Potential
            UseDispSponge = False,  # Dispersive Boundary Condition
            SelfGravity = True,
            NBodyInterp = True,
            NBodyGravity = True,
            Shift = False,          # Shift NBody Location Between Grids (For QW)
            Simpson = False,        # Not Used.
            Silent = True,         # Quiet Mode
            AutoStop = False,       # Stop with BH
            AutoStop2 = False,      # Stop with Gravitational Field Strength
            WellThreshold = 100,    # Stop Condition
            InitPath = '',          # Everything except for the .npy file extension.
            InitWeight = 1,
            Stream = False,          # Write RK Step Results To File
            StreamChar = [])     # Locations in the vectorized TMState to Stream. (x-y flipped).

Duration = time.time()-Start_Time

sim_number = PyUL.ULDStepEst(duration,duration_units,
                                          length,length_units,
                                          resol,step_factor, 
                                          save_number = -1)

DPS = Duration/sim_number

printU('If desired, please copy and send the following text to the developer (as a GitHub ticket or via Email)','Bench')

print("="*40+'PyUL Benchmark Result')

uname = platform.uname()

print(f"System: {uname.system}")
print(f"Release: {uname.release}")
print(f"Version: {uname.version}")
print(f"Machine: {uname.machine}")
print(f"Processor: {uname.processor}")

Threads, ThreadSource = PyUL._compat.resolve_num_threads(resol)
print(f"\nAvailable CPU Threads: {os.cpu_count()} (FFT threads used: {Threads}, {ThreadSource})")
print(f"FFT backend: {PyUL._compat.resolve_fft_backend().name}")
print(f"Resolution: {resol}")
print(f"Avg Step (s): {DPS:.3g}")
print("="*80)
