## Notes

### 2025-05-19

Finalized the calibration tests

### 2025-05-20

Today I am going to test the stochastic ALP field where $\tau_a \ll T_2$.

With the data, $\sqrt{\tau_a T_2}$ and $\sqrt{\tau_a \Delta T}$ are proven to be true for estimating tipping angle.

### 2025-05-26

##### Quantitive analysis of the M_transvers over time

$M_t = M_0 \, \sin\theta \approx  M_0 \,\theta$ assuming $\theta\ll1$. We have $M_0=1$, and

$\theta=\Omega \Delta t = \gamma B_a \Delta t$.

The keypoint here is the strength of the magnetic field $B_a$.


### 2025-06-02

#### Study the effect of the frequency detuning on the tipping angle

Using $\tau_a \ll T_2$.

I used 40 $\nu_a$'s and wanted to see the tipping angle evolution over time. It is like scanning over the frequencies.

Findings:

1. When on resonance ($\nu_a\approx\nu_\mathrm{Larmor}$), the tipping angle growes up first ($t\ll T_2 \text{ or } \tau_a$), then saturates after $t\gg T_2 \text{ or } \tau_a$.

2. When off resonance ($|\nu_a-\nu_\mathrm{Larmor}|\gg \Delta\nu_a$), the tipping angle growes up first ($t\ll T_2 \text{ or } \tau_a$), then declines by a small amount, and becomes steady after $t\gg T_2 \text{ or } \tau_a$.

The data and figures are in the folder /20250602-tau_a_《_T2.

One of the conclusions: the difference between $\nu_a$ and $\nu_\mathrm{Larmor}$ matters in the magnitude of the tipping angle.

### 2025-06-03

what to do today...

I can use the lineshape of the tipping angle magnitude to quantitvely evaluate the effect of axion field. It should be something like this:

$\theta^2 \propto \lambda(\nu)$

or more precisely:

$\theta^2 \approx (\gamma B_a)^2\, \lambda(\nu)$ nono this is not right. check its unit!

### 2025-08-08

#### magnitude test

How to quantitvely evaluate the effect of axion field? Here I do not consider the T2*, but only detuning $\Delta\nu$, $\tau_a$, and $T_2$. Assuming measurement time much longer than any one of the times above.

We expected

$\theta^2 \approx (\gamma B_a T_2)^2\, \lambda(\nu)$ when $T_2 \ll \tau_a$

$\theta^2 \approx^? (\gamma B_a )^2 \tau_a T_2\, \lambda(\nu)$ when $\tau_a\ll T_2\ll T_\mathrm{meas}$.

Let us verify these.

### 2025-09-19

#### Brms magnitude test

in [src\tests\20250919-Bamp-Brms-fft-ifft-magnitude-test\check_B_rms.py] we test the relationship between Bamp (input axion Ba field amplitude) and the output Brms (rms of the simulated axion B field).

In principle Brms is equal to Bamp. However, due to numpy.fft and numpy.ifft, the amplitude of simulated Ba field should be Bamp $\times$ some N, where N is simulation rate or array length.

Test 0:
------------------------
(input) Bamp = 1e-10

(output) np.mean(B_rms_from_simu_arr) : float64(3.0370544560166723e-12)

np.std(B_rms_from_simu_arr) : float64(3.161938491415789e-13)

simuRate * duration = 500 * 20 = 10000

--------------------
np.mean(B_rms_from_simu_arr) : float64(2.7800346877931807e-12)

np.std(B_rms_from_simu_arr) : float64(8.35364629679566e-13)

simuRate * duration = 500 * 2 = 1000

---------------------------------------
np.mean(B_rms_from_simu_arr) : float64(2.733154499609955e-12)

np.std(B_rms_from_simu_arr) : float64(1.2322121825485608e-12)

simuRate * duration = 500 * 2 = 1000

---------------------

np.mean(B_rms_from_simu_arr) : float64(1.0460985972196068e-11)

np.std(B_rms_from_simu_arr) : float64(3.5039550658350867e-12)

simuRate * duration = 50 * 2 = 100

--------------

line: ax_FFT = Bamp * ax_lineshape * rvs_phase * **simuRate * np.sqrt(duration)**

fixed the problem.

Note that Brms of Bx or By is = Bamp / np.sqrt(2)

### 2025-10-21

#### Plan

I am going to creatre a few classes for better clarifying the simulation conditions, including:

1. SQUID
2. Pickup
3. Sample
4. AxionWind
5. MagField
6. Simulation
7. Experiment: LF or HF -> give fluxpower, exclusions, etc.
8.
