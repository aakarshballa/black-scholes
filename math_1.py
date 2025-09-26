import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / (math.sqrt(2 * math.pi))

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))\

def d1_d2 (S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("All inputs must be positive and T, sigma must be greater than 0.")
    d1 = (math.log(S/K) + (r - 1 + sigma * sigma / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - (sigma * math.sqrt(T))
    return d1, d2

def bs_price(S, K, T, r, q, sigma, option_type='call'):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    if (math.isnan(d1) or math.isnan(d2)):
        raise ValueError("Invalid inputs, resulting in NaN values.")
    if option_type == 'call':
        price = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

def greeks(S, K, T, r, q, sigma, option_type="call"):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    
    # Probability density function of d1
    pdf_d1 = norm_pdf(d1)
    # Cumuluative distribution probabilities of d1 and 2
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    # Same as e ^ (-qT) / e ^ (-rT)
    disc_q = math.exp(-q*T)    # dividend discount factor
    disc_r = math.exp(-r*T)    # risk-free discount factor
    sqrtT = math.sqrt(T)

    # Gamma is the same for calls and puts
    # Sensitivity of Delta to changes in the underlying asset price
    gamma = (math.exp(-q * T) * pdf_d1) / (S * sigma * sqrtT)
    # Vega is also the same for calls and puts
    # Sensitivity of option price to change in volatility
    vega = S * disc_q * pdf_d1 * sqrtT

    # Delta is the sensitivity in option price to changes in the underlying asset price
    # Delta is positive for calls and negative for puts
    # Theta is the sensitivity in option price to the passage of time
    # Rho is the sensitivity in option price to changes in the risk-free interest rate
    if option_type == "call":
        delta = disc_q * Nd1
        theta = -0.5 * (S * disc_q * pdf_d1 * sigma) / sqrtT + q * S * disc_q * Nd1 - r * K * disc_r * Nd2
        rho = K * T * disc_r * Nd2
    elif option_type == "put":
        delta = disc_q * (Nd1 - 1)
        theta = -0.5 * (S * disc_q * pdf_d1 * sigma) / sqrtT - q * S * disc_q * (1 - Nd1) + r * K * disc_r * (1 - Nd2)
        rho = -K * T * disc_r * (1 - Nd2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,   # per 1% change in volatility
        "theta": theta / 365, # per day
        "rho": rho / 100      # per 1% change in rate
    }


def moneyness_label (S, K, option_type="call", tol=0.01):
    if (abs(S/K - 1.0) <= tol):
        return "ATM"
    elif (S/K > 1.0 + tol):
        return "ITM" if option_type == "call" else "OTM"
    else:
        return "OTM" if option_type == "call" else "ITM"

def build_grid(S, r, q, sigma, T, opt_type="call",
               m_lo=0.7, m_hi=1.3, step=2.5, atm_tol=0.01):
    
    # Will generate strikes from max(0.01, S*m_lo) to S*m_hi with step size of 'step'
    K_min = max(0.01, S * m_lo)
    K_max = S * m_hi
    strikes = np.arange(K_min, K_max + step, step)

    rows = []
    for K in strikes:
        price = bs_price(S, K, r, q, sigma, T, opt_type)
        g = greeks(S, K, r, q, sigma, T, opt_type)
        label = moneyness_label(S, K, opt_type, tol=atm_tol)
        rows.append({
            "Strike (K)": round(float(K), 2),
            "K/S": K / S,
            "Moneyness": label,
            "BS Price": price,
            "Delta": g["delta"],
            "Gamma": g["gamma"],
            "Vega": g["vega"],
            "Theta (per year)": g["theta"],
            "Rho": g["rho"],
        })
    df = pd.DataFrame(rows)
    return df

def plot_price_vs_strike(df):
    fig, ax = plt.subplots()
    ax.plot(df["Strike (K)"], df["BS Price"])
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Price")
    ax.set_title("BS Price vs Strike")
    fig.tight_layout()
    return fig

def plot_greek_vs_strike(df, greek="Delta"):
    if greek not in df.columns:
        raise ValueError(f"Greek {greek} not found in df.")
    fig, ax = plt.subplots()
    ax.plot(df["Strike (K)"], df[greek])
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel(greek)
    ax.set_title(f"{greek} vs Strike")
    fig.tight_layout()
    return fig

S, r, q, sigma, T = 100, 0.035, 0.00, 0.25, 0.5
df = build_grid(S, r, q, sigma, T, "call", 0.7, 1.3, 2.5)
plot_price_vs_strike(df)
plot_greek_vs_strike(df, "Delta")
plt.show()