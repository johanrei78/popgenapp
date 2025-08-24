# -*- coding: utf-8 -*-
"""
Populasjonsgenetikk-simulator
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# ------------------ Simuleringsfunksjoner ------------------

def simulate_one_pop(p0, N, wAA, wAa, waa, mu, nu, generations):
    """Simuler én populasjon (med eller uten drift)."""
    p = p0
    freqs = [p0]

    for _ in range(generations):
        p2 = p**2
        pq = 2*p*(1-p)
        q2 = (1-p)**2

        w_bar = p2*wAA + pq*wAa + q2*waa
        if w_bar == 0:
            break
        p_prime = (p2*wAA + 0.5*pq*wAa) / w_bar

        # Mutasjon
        p_prime = p_prime*(1 - mu) + (1 - p_prime)*nu

        # Drift eller ikke
        if N is not None:
            p = np.random.binomial(2*N, p_prime) / (2*N)
        else:
            p = p_prime

        freqs.append(np.clip(p, 0, 1))
    return np.array(freqs)


def simulate_two_pops(p0_1, p0_2, N, fitness1, fitness2, mu, nu,
                      generations, migrate=False, m12=0, m21=0):
    """Simuler to populasjoner, med mulighet for migrasjon."""
    p = np.array([p0_1, p0_2])
    freqs = [p.copy()]

    for _ in range(generations):
        new_p = []
        for i, pi in enumerate(p):
            wAA, wAa, waa = fitness1 if i == 0 else fitness2

            p2 = pi**2
            pq = 2*pi*(1-pi)
            q2 = (1-pi)**2

            w_bar = p2*wAA + pq*wAa + q2*waa
            if w_bar == 0:
                p_prime = pi
            else:
                p_prime = (p2*wAA + 0.5*pq*wAa) / w_bar

            # Mutasjon
            p_prime = p_prime*(1 - mu) + (1 - p_prime)*nu

            # Drift eller ikke
            if N is not None:
                pi_next = np.random.binomial(2*N, p_prime) / (2*N)
            else:
                pi_next = p_prime

            new_p.append(pi_next)

        p = np.array(new_p)

        # Migrasjon
        if migrate:
            p1 = (1-m21)*p[0] + m21*p[1]
            p2 = (1-m12)*p[1] + m12*p[0]
            p = np.array([p1, p2])

        freqs.append(np.clip(p.copy(), 0, 1))
    return np.array(freqs)


# ------------------ Streamlit UI ------------------

st.title("Populasjonsgenetikk-simulator")

# Modellvalg
pop_mode = st.radio("Populasjonsmodell:",
                    ["Begrenset populasjonsstørrelse", "Uendelig stor populasjon"])

if pop_mode == "Begrenset populasjonsstørrelse":
    N = st.number_input("Populasjonsstørrelse (N)",
                        min_value=10, max_value=10000, value=100)
else:
    N = None  # signaliserer uendelig populasjon

generations = st.slider("Antall generasjoner", 10, 500, 100)
num_pops = st.radio("Antall populasjoner:", [1, 2])

st.markdown("### Mutasjonsrate")
mu = st.number_input("Mutasjonsrate A → a", 0.0, 1.0, 0.0,
                     step=0.0001, format="%.4f")
nu = st.number_input("Mutasjonsrate a → A", 0.0, 1.0, 0.0,
                     step=0.0001, format="%.4f")

# --- Én populasjon ---
if num_pops == 1:
    st.markdown("### Startfrekvens til allel A")
    p0 = st.slider("Startfrekvens til allel A", 0.0, 1.0, 0.5, step=0.01, label_visibility="collapsed")

    st.markdown("### Fitness til genotyper")
    wAA = st.slider("Fitness til AA", 0.0, 2.0, 1.0, step=0.01)
    wAa = st.slider("Fitness til Aa", 0.0, 2.0, 1.0, step=0.01)
    waa = st.slider("Fitness til aa", 0.0, 2.0, 1.0, step=0.01)

    if st.button("Kjør simulering"):
        freqs = simulate_one_pop(p0, N, wAA, wAa, waa,
                                 mu, nu, generations)
        fig, ax = plt.subplots()
        ax.plot(range(len(freqs)), freqs, label="Populasjon 1", color='blue')
        ax.set_xlabel("Generasjon")
        ax.set_ylabel("Frekvens av A")
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

# --- To populasjoner ---
else:
    st.markdown("### Startfrekvens til allel A")
    p0_1 = st.slider("Populasjon 1", 0.0, 1.0, 0.5, step=0.01)
    p0_2 = st.slider("Populasjon 2", 0.0, 1.0, 0.5, step=0.01)

    st.markdown("### Fitness til genotyper i populasjon 1")
    wAA1 = st.slider("Fitness til AA", 0.0, 2.0, 1.0, step=0.01, key="wAA1")
    wAa1 = st.slider("Fitness til Aa", 0.0, 2.0, 1.0, step=0.01, key="wAa1")
    waa1 = st.slider("Fitness til aa", 0.0, 2.0, 1.0, step=0.01, key="waa1")

    st.markdown("### Fitness til genotyper i populasjon 2")
    wAA2 = st.slider("Fitness til AA", 0.0, 2.0, 1.0, step=0.01, key="wAA2")
    wAa2 = st.slider("Fitness til Aa", 0.0, 2.0, 1.0, step=0.01, key="wAa2")
    waa2 = st.slider("Fitness til aa", 0.0, 2.0, 1.0, step=0.01, key="waa2")

    migrate = st.checkbox("Inkluder migrasjon mellom populasjoner", value=False)
    if migrate:
        m12 = st.slider("Migrasjonsrate fra populasjon 1 → 2", 0.0, 1.0, 0.0, step=0.01)
        m21 = st.slider("Migrasjonsrate fra populasjon 2 → 1", 0.0, 1.0, 0.0, step=0.01)
    else:
        m12 = m21 = 0.0

    if st.button("Kjør simulering"):
        freqs = simulate_two_pops(p0_1, p0_2, N,
                                  (wAA1, wAa1, waa1),
                                  (wAA2, wAa2, waa2),
                                  mu, nu, generations,
                                  migrate, m12, m21)
        fig, ax = plt.subplots()
        ax.plot(range(len(freqs)), freqs[:, 0], label="Populasjon 1", color='blue')
        ax.plot(range(len(freqs)), freqs[:, 1], label="Populasjon 2", color='red')
        ax.set_xlabel("Generasjon")
        ax.set_ylabel("Frekvens av A")
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
