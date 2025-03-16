import streamlit as st
import math
import numpy as np
from scipy import stats
import plotly.express as px
import pandas as pd

def calculate_power(n_total, treatment_pct, mde, alpha=0.05):
    """Calculate statistical power given total sample size, treatment percentage, and MDE."""
    # Calculate group sizes
    n1 = int(n_total * treatment_pct / 100)  # Treatment group
    n2 = n_total - n1  # Control group
    
    # Ensure we don't have empty groups
    if n1 == 0 or n2 == 0:
        return 0, n1, n2
    
    # Pooled proportion (assuming equal proportions in null hypothesis)
    p_pooled = 0.5  # Using 0.5 as a default which provides the most conservative estimate
    
    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Critical value
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Power
    power = 1 - stats.norm.cdf(z_alpha - mde/se) + stats.norm.cdf(-z_alpha - mde/se)
    return power, n1, n2

def calculate_mde(n_total, treatment_pct, power, alpha=0.05):
    """Calculate minimum detectable effect given total sample size, treatment percentage, and power."""
    # Calculate group sizes
    n1 = int(n_total * treatment_pct / 100)  # Treatment group
    n2 = n_total - n1  # Control group
    
    # Ensure we don't have empty groups
    if n1 == 0 or n2 == 0:
        return 1, n1, n2
    
    # Pooled proportion
    p_pooled = 0.5
    
    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # MDE
    mde = (z_alpha + z_beta) * se
    return mde, n1, n2

def calculate_sample_size(mde, power, treatment_pct, alpha=0.05):
    """
    Calculate required sample size given MDE, power, treatment percentage, and alpha.
    Modified to ensure consistency with power and MDE calculations.
    """
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Pooled proportion
    p_pooled = 0.5
    
    # Treatment proportion
    treatment_prop = treatment_pct / 100
    
    # Calculate total sample size needed
    term1 = (z_alpha + z_beta)**2
    term2 = p_pooled * (1 - p_pooled)
    term3 = 1 / (treatment_prop * (1 - treatment_prop))
    n_total = term1 * term2 * term3 / (mde**2)
    
    # Calculate individual group sizes
    n1 = round(n_total * treatment_prop)  # Treatment group
    n2 = round(n_total * (1 - treatment_prop))  # Control group
    n_total = n1 + n2  # Recalculate total based on rounded group sizes
    
    return n_total, n1, n2

def generate_treatment_comparison_data(calculation_type, fixed_params):
    """Generate data for treatment percentage comparison graphs."""
    treatment_percentages = list(range(5, 96, 5))
    results = []
    
    if calculation_type == "Sample Size":
        # Fixed: MDE, power, alpha
        mde = fixed_params["mde"]
        power = fixed_params["power"]
        alpha = fixed_params["alpha"]
        
        for tp in treatment_percentages:
            n_total, n1, n2 = calculate_sample_size(mde/100, power, tp, alpha)
            results.append({
                "treatment_pct": tp, 
                "sample_size": n_total, 
                "treatment_size": n1, 
                "control_size": n2
            })
    
    elif calculation_type == "MDE":
        # Fixed: n_total, power, alpha
        n_total = fixed_params["n_total"]
        power = fixed_params["power"]
        alpha = fixed_params["alpha"]
        
        for tp in treatment_percentages:
            mde, n1, n2 = calculate_mde(n_total, tp, power, alpha)
            results.append({
                "treatment_pct": tp, 
                "mde": mde*100,  # Convert to percentage points
                "treatment_size": n1, 
                "control_size": n2
            })
    
    return pd.DataFrame(results)

def display_summary(n_total, treatment_pct, n1, n2, mde, power, alpha):
    """Display summary statistics in a nice format."""
    st.markdown("### 📋 Summary")
    
    # Create colored boxes for treatment and control groups
    st.markdown(f"""
    <div class="treatment-control-display">
        <div class="treatment-box" style="width:{treatment_pct}%">
            Treatment: {treatment_pct:.1f}% ({n1})
        </div>
        <div class="control-box" style="width:{100-treatment_pct}%">
            Control: {100-treatment_pct:.1f}% ({n2})
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary table
    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | Total Sample Size | {n_total} |
    | Treatment Group | {n1} ({treatment_pct:.1f}%) |
    | Control Group | {n2} ({100-treatment_pct:.1f}%) |
    | Minimum Detectable Effect | {mde:.2f} percentage points |
    | Statistical Power | {power:.4f} |
    | Significance Level (α) | {alpha} |
    """)

def main():
    # Set page config with a modern theme
    st.set_page_config(
        page_title="Power Calculator ⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for a more modern look
    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: 0.5px;
    }
    .stProgress > div > div {
        background-image: linear-gradient(to right, #28a745, #dc3545);
    }
    .treatment-control-display {
        display: flex;
        align-items: center;
        margin-top: 0.5rem;
    }
    .treatment-box {
        background-color: #28a745;
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
        margin-right: 0.5rem;
        flex-grow: 1;
    }
    .control-box {
        background-color: #dc3545;
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
        flex-grow: 1;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("✨ Statistical Power Calculator")
    st.markdown("""
    Interactive tool for A/B test power calculations. Adjust parameters to see how they affect each other.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Calculator", "📈 Treatment % Analysis", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.header("🎛️ Parameters")
            calculation_type = st.radio(
                "What would you like to calculate?",
                ["🔋 Power", "🎯 Minimum Detectable Effect (MDE)", "👥 Sample Size"]
            )
            
            alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
            
            # Only show MDE example when not calculating MDE
            if calculation_type != "🎯 Minimum Detectable Effect (MDE)":
                st.markdown("### 🔍 MDE Example")
                control_rate = st.number_input("Control Group Rate (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
                treatment_rate = st.number_input("Treatment Group Rate (%)", min_value=0.0, max_value=100.0, value=50.1, step=0.1)
                example_mde = abs(treatment_rate - control_rate)
                st.markdown(f"**Example MDE: {example_mde:.2f} percentage points**")
                st.markdown(f"This means detecting a difference of {example_mde:.2f} percentage points between groups.")
                
                st.markdown("---")
            
            if calculation_type == "🔋 Power":
                n_total = st.number_input("Total Sample Size", min_value=20, value=2000, step=10)
                treatment_pct = st.slider("Treatment Group Percentage (%)", 1.0, 99.0, 50.0, 0.1)
                mde_default = example_mde if calculation_type != "🎯 Minimum Detectable Effect (MDE)" else 0.1
                mde = st.slider("Minimum Detectable Effect (percentage points)", 0.01, 20.0, mde_default, 0.01)
                
                if st.button("💪 Calculate Power", use_container_width=True):
                    power, n1, n2 = calculate_power(n_total, treatment_pct, mde/100, alpha)  # Convert percentage points to proportion
                    
                    with col2:
                        st.header("🔍 Results")
                        
                        # Power result with emoji
                        power_emoji = "🚀" if power >= 0.8 else "🔋"
                        st.markdown(f"### {power_emoji} Power: {power:.4f}")
                        st.progress(min(power, 1.0))
                        
                        display_summary(n_total, treatment_pct, n1, n2, mde, power, alpha)
                        
            elif calculation_type == "🎯 Minimum Detectable Effect (MDE)":
                n_total = st.number_input("Total Sample Size", min_value=20, value=2000, step=10)
                treatment_pct = st.slider("Treatment Group Percentage (%)", 1.0, 99.0, 50.0, 0.1)
                power = st.slider("Statistical Power", 0.5, 0.99, 0.8, 0.01)
                
                if st.button("🎯 Calculate MDE", use_container_width=True):
                    mde, n1, n2 = calculate_mde(n_total, treatment_pct, power, alpha)
                    mde_pct = mde * 100  # Convert proportion to percentage points
                    
                    with col2:
                        st.header("🔍 Results")
                        st.markdown(f"### 🎯 Minimum Detectable Effect: {mde_pct:.2f} percentage points")
                        
                        display_summary(n_total, treatment_pct, n1, n2, mde_pct, power, alpha)
                        
            elif calculation_type == "👥 Sample Size":
                mde_default = example_mde if 'example_mde' in locals() else 0.1
                mde = st.slider("Minimum Detectable Effect (percentage points)", 0.01, 20.0, mde_default, 0.01)
                power = st.slider("Statistical Power", 0.5, 0.99, 0.8, 0.01)
                treatment_pct = st.slider("Treatment Group Percentage (%)", 1.0, 99.0, 50.0, 0.1)
                
                if st.button("👥 Calculate Sample Size", use_container_width=True):
                    n_total, n1, n2 = calculate_sample_size(mde/100, power, treatment_pct, alpha)  # Convert percentage points to proportion
                    
                    with col2:
                        st.header("🔍 Results")
                        st.markdown(f"### 👥 Required Sample Sizes")
                        st.markdown(f"Treatment Group: {n1} ({treatment_pct:.1f}%)")
                        st.markdown(f"Control Group: {n2} ({100-treatment_pct:.1f}%)")
                        st.markdown(f"Total Sample Size: {n_total}")
                        
                        display_summary(n_total, treatment_pct, n1, n2, mde, power, alpha)
    
    with tab2:
        st.header("📊 Treatment Percentage Analysis")
        st.markdown("See how different treatment/control splits affect your results.")
        
        analysis_type = st.radio(
            "What would you like to analyze?", 
            ["Sample Size vs Treatment %", "MDE vs Treatment %"]
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Parameters")
            if analysis_type == "Sample Size vs Treatment %":
                analysis_mde = st.slider("MDE (percentage points)", 0.01, 10.0, 0.1, 0.01, key="analysis_mde")
                analysis_power = st.slider("Power", 0.5, 0.99, 0.8, 0.01, key="analysis_power")
                analysis_alpha = st.slider("Alpha", 0.01, 0.1, 0.05, 0.01, key="analysis_alpha")
                
                fixed_params = {
                    "mde": analysis_mde,
                    "power": analysis_power,
                    "alpha": analysis_alpha
                }
                
                comparison_data = generate_treatment_comparison_data("Sample Size", fixed_params)
                
                with col2:
                    fig = px.line(
                        comparison_data, 
                        x="treatment_pct", 
                        y="sample_size",
                        title="📊 Total Sample Size vs Treatment Percentage",
                        labels={
                            "treatment_pct": "Treatment Group Percentage (%)",
                            "sample_size": "Required Total Sample Size"
                        },
                        custom_data=["treatment_size", "control_size"]
                    )
                    
                    # Add hover template
                    fig.update_traces(
                        hovertemplate="<b>Treatment:</b> %{x}%<br>" +
                        "<b>Total Sample Size:</b> %{y}<br>" +
                        "<b>Treatment Group:</b> %{customdata[0]}<br>" +
                        "<b>Control Group:</b> %{customdata[1]}"
                    )
                    
                    fig.update_layout(
                        hovermode="x unified",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
# Add a note about the optimal point
                    min_idx = comparison_data["sample_size"].idxmin()
                    opt_treatment = comparison_data.loc[min_idx, "treatment_pct"]
                    opt_sample = comparison_data.loc[min_idx, "sample_size"]
                    st.info(f"📌 The optimal treatment percentage is around {opt_treatment}%, which requires a minimum sample size of {opt_sample}.")
                    
            else:  # MDE vs Treatment %
                analysis_n_total = st.number_input("Total Sample Size", min_value=100, value=2000, step=100, key="analysis_n_total")
                analysis_power = st.slider("Power", 0.5, 0.99, 0.8, 0.01, key="analysis_power2")
                analysis_alpha = st.slider("Alpha", 0.01, 0.1, 0.05, 0.01, key="analysis_alpha2")
                
                fixed_params = {
                    "n_total": analysis_n_total,
                    "power": analysis_power,
                    "alpha": analysis_alpha
                }
                
                comparison_data = generate_treatment_comparison_data("MDE", fixed_params)
                
                with col2:
                    fig = px.line(
                        comparison_data, 
                        x="treatment_pct", 
                        y="mde",
                        title="📏 MDE vs Treatment Percentage",
                        labels={
                            "treatment_pct": "Treatment Group Percentage (%)",
                            "mde": "Minimum Detectable Effect (percentage points)"
                        },
                        custom_data=["treatment_size", "control_size"]
                    )
                    
                    # Add hover template
                    fig.update_traces(
                        hovertemplate="<b>Treatment:</b> %{x}%<br>" +
                        "<b>MDE:</b> %{y:.4f} percentage points<br>" +
                        "<b>Treatment Group:</b> %{customdata[0]}<br>" +
                        "<b>Control Group:</b> %{customdata[1]}"
                    )
                    
                    fig.update_layout(
                        hovermode="x unified",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a note about the optimal point
                    min_idx = comparison_data["mde"].idxmin()
                    opt_treatment = comparison_data.loc[min_idx, "treatment_pct"]
                    opt_mde = comparison_data.loc[min_idx, "mde"]
                    st.info(f"📌 The optimal treatment percentage is around {opt_treatment}%, which gives the minimum MDE of {opt_mde:.4f} percentage points.")
    
    with tab3:
        st.header("ℹ️ About this Calculator")
        
        st.markdown("""
        ### 🧮 What is Statistical Power?
        
        Statistical power is the probability that a test correctly rejects the null hypothesis when it is false. In A/B testing, it's your ability to detect a real effect when it exists.
        
        ### 🎯 What is MDE?
        
        Minimum Detectable Effect (MDE) is the smallest effect size that your test can reliably detect given your sample size, significance level, and desired power. It's expressed in percentage points in this calculator.
        
        ### 📋 Terminology
        
        - **Treatment Group**: The group receiving the new experience or variant
        - **Control Group**: The group receiving the standard experience
        - **Alpha (α)**: The significance level, or probability of Type I error (false positive)
        - **Power (1-β)**: The probability of detecting an effect when it exists
        - **Sample Size**: The number of observations needed in your experiment
        
        ### 💡 Tips for Effective A/B Testing
        
        - 50/50 splits aren't always optimal - use the Treatment % Analysis tab to find the most efficient split
        - Consider practical significance, not just statistical significance
        - Run tests long enough to account for time-based factors
        - Ensure random assignment to treatment and control groups
        """)

if __name__ == "__main__":
    main()