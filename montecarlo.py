from math import exp, log, sqrt
import datetime as dt
import numpy as np
from opciones import Analysis
import matplotlib.pyplot as plt 

### FUNCIONES

def montecarlo_binominal(s0, strike, maturity, sigma, r, n, sims):
    """
    Calculates the price of an option call by introducing a set of parameters. 
    - s0: Underlying price at t0.
    - strike: Contract value of the call option.
    - maturity: period of time until the expiration of the contract in annual terms.
    - sigma: estimated constant volatility in annual terms.
    - r: estimated constant interest rate in annual terms.
    - sims: number of M simulations of the payoff of the call option.
    """
    delta = maturity / n
    m = r - sigma**2 / 2
    u = exp(m * delta + sigma * sqrt(delta))
    d = exp(m * delta - sigma * sqrt(delta))
    R = exp(r*delta)
    p = (R-d) / (u-d)   

    sims_vector = []
    payoff_final = []
    for w in range(sims):
        ran = np.random.rand(n)
        sims_vector.append(ran)

        log_sims = []
        for sim in ran:
            if sim < p:
                ret = log(d)
                log_sims.append(ret)
            elif sim > p:
                ret = log(u)
                log_sims.append(ret)
        suma = s0 * exp(sum(log_sims))

        po = max((suma-strike),0) 
        payoff_final.append(po)

    call_mc = np.mean(payoff_final) * exp(-r*maturity)
    return call_mc

def montecarlo_difusion(s0,strike,maturity,sigma, r, n, sims):
    """
    Calculates the price of an option call by introducing a set of parameters. 
    - s0: Underlying price at t0.
    - strike: Contract value of the call option.
    - maturity: period of time until the expiration of the contract in annual terms.
    - sigma: estimated constant volatility in annual terms.
    - r: estimated constant interest rate in annual terms.
    - sims: number of M simulations of the payoff of the call option.
    """
    m = r - sigma**2 / 2
    delta = maturity/n

    def create_sim(s0,r,sigma,n):
        st = s0
        def generate_value():
            nonlocal st
            for i in range(n):
                st *= exp(m * delta + sigma * sqrt(delta) * np.random.randn())
            return st
        return generate_value()

    sim_values = []
    pf_final = []
    for i in range(sims):
        st = create_sim(s0,r,sigma,n)
        sim_values.append(st)
        pf = max(st - strike, 0) 
        pf_final.append(pf)

    call_mc_2 = np.mean(pf_final) * exp(-r * maturity)
    
    return call_mc_2

if __name__ == '__main__':
    ### PARAMS 
    s0 = 100
    strike = 100
    maturity = 0.50
    sigma = 0.3
    r = 0.05
    n = 100
    simulations = 100000
    today = dt.datetime.today()

    ### BLACK & SCHOLES CALCULATION
    test_analysis = Analysis(s0,maturity,today,maturity,r,sigma)
    call_bs = test_analysis.black_scholes(s0,strike,maturity,r,sigma)[2]

    ### MONTECARLO BINOMIAL CALCULATION
    delta = maturity/n
    m = r - sigma**2 / 2
    u = exp(m * delta + sigma * sqrt(delta))
    d = exp(m * delta - sigma * sqrt(delta))
    R = exp(r*delta)
    p = (R-d) / (u-d)

    sims_vector = []
    payoff_final = []
    for w in range(simulations):
        ran = np.random.rand(n)
        sims_vector.append(ran)

        log_sims = []
        for sim in ran:
            if sim < p:
                ret = log(d)
                log_sims.append(ret)
            elif sim > p:
                ret = log(u)
                log_sims.append(ret)
        suma = s0 * exp(sum(log_sims))

        po = max((suma-strike),0) * exp(-r*maturity)
        payoff_final.append(po)

    call_mc = np.mean(payoff_final)
    st_call_mc = np.std(payoff_final) / sqrt(simulations)

    ### MONTECARLO NORMAL CALCULATION
    def create_sim(s0,r,sigma,n):
        st = s0
        def generate_value():
            nonlocal st  
            for i in range(n):
                st *= exp(m * delta + sigma * sqrt(delta) * np.random.randn())
            return st
        return generate_value()

    sim_values = []
    pf_final = []
    for i in range(simulations):
        st = create_sim(s0,r,sigma,n)
        sim_values.append(st)
        pf = max(st - strike, 0) * exp(-r * maturity)
        pf_final.append(pf)

    call_mc_2 = np.mean(pf_final)

    print(f'Valuacion por Montecarlo binominal: {round(call_mc,4)}\nValuacion por Montecarlo difusion: {call_mc_2}\nValuacion por BS: {round(call_bs,4)}')
