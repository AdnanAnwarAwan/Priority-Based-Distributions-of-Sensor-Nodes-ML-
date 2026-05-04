"""
energy_model.py — First-order radio energy model (Heinzelman et al., 2000).

E_tx(k,d) = k·E_elec + k·ε_fs·d²    (d < d_thresh ≈ 87 m)
E_tx(k,d) = k·E_elec + k·ε_mp·d⁴    (d ≥ d_thresh)
E_rx(k)   = k·E_elec
E_sleep   ≈ 0.001 × E_active  (hibernation saves ~99.9%)
"""
from __future__ import annotations
import numpy as np


class EnergyModel:
    def __init__(
        self,
        e_elec:  float = 50e-9,
        eps_fs:  float = 10e-12,
        eps_mp:  float = 0.0013e-12,
        e_da:    float = 5e-9,
        e_init:  float = 0.5,
        e_sleep_frac: float = 0.001,
    ):
        self.e_elec       = e_elec
        self.eps_fs       = eps_fs
        self.eps_mp       = eps_mp
        self.e_da         = e_da
        self.initial      = e_init
        self.e_sleep_frac = e_sleep_frac
        self.d_thresh     = np.sqrt(eps_fs / eps_mp)   # ≈ 87.7 m

    def tx_energy(self, bits: int, distance: float) -> float:
        if distance < self.d_thresh:
            return bits * (self.e_elec + self.eps_fs * distance**2)
        return bits * (self.e_elec + self.eps_mp * distance**4)

    def rx_energy(self, bits: int) -> float:
        return bits * self.e_elec

    def agg_energy(self, bits: int) -> float:
        return bits * self.e_da

    def sleep_energy(self, active_cost: float) -> float:
        return active_cost * self.e_sleep_frac

    def round_energy(self, bits, is_ch, n_members, dist_to_sink, dist_to_ch=0.0):
        if is_ch:
            return (self.rx_energy(bits * n_members)
                    + self.agg_energy(bits * (n_members + 1))
                    + self.tx_energy(bits, dist_to_sink))
        return self.tx_energy(bits, dist_to_ch)

    def lifetime_rounds(self, n_nodes=100, bits=4000, area=100.0, p_ch=0.05) -> int:
        d_avg = area / (2 * np.sqrt(np.pi * p_ch))
        d_bs  = 0.765 * area / 2
        e_per = ((1 - p_ch) * self.tx_energy(bits, d_avg)
                 + p_ch * (self.rx_energy(bits / p_ch)
                            + self.agg_energy(bits / p_ch)
                            + self.tx_energy(bits, d_bs)))
        return int(self.initial / e_per)
