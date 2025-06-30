import matplotlib.pyplot as plt
import numpy as np
from deep_learning_models import compare_models

def show_graphs():
    # Get results from model comparison
    results = compare_models("dataset", epochs=5, num_images=3)
    
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
    
    # Define colors with emphasis on hybrid model
    colors = {
        'hybrid': '#FF4B4B',  # Bright red for hybrid
        'drn': '#B0B0B0',     # Light gray for DRN
        'vit': '#B0B0B0',     # Light gray for ViT
        'traditional': '#B0B0B0'  # Light gray for traditional
    }
    
    # Plot 1: IRR Line Plot
    distortion_params = np.linspace(0.0, 0.8, 5)
    ax1.plot(distortion_params, results['hybrid_irr_trend'], 
             'r-', label='Hybrid', linewidth=3, color=colors['hybrid'])
    ax1.plot(distortion_params, results['drn_irr_trend'], 
             'b-', label='DRN', linewidth=2, color=colors['drn'], alpha=0.7)
    ax1.plot(distortion_params, results['vit_irr_trend'], 
             'g-', label='ViT', linewidth=2, color=colors['vit'], alpha=0.7)
    ax1.plot(distortion_params, results['existing_gpass_irr_trend'], 
             'k--', label='Traditional GPass', linewidth=2, color=colors['traditional'], alpha=0.7)
    
    # Add annotation for hybrid model's best performance
    ax1.annotate('Best Performance', 
                xy=(0.4, results['hybrid_irr_trend'][2]), 
                xytext=(0.2, 0.7),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    ax1.set_xlabel('Distortion Parameter', fontsize=12)
    ax1.set_ylabel('IRR', fontsize=12)
    ax1.set_title('(a) Information Retention Rate (IRR)', fontsize=14, pad=15)
    ax1.grid(True, which='both', linestyle='-', alpha=0.2)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_ylim(0.2, 0.9)
    ax1.set_xlim(0.0, 0.8)
    
    # Plot 2: PDS Line Plot
    ax2.plot(distortion_params, results['hybrid_pds_trend'], 
             'r-', label='Hybrid', linewidth=3, color=colors['hybrid'])
    ax2.plot(distortion_params, results['drn_pds_trend'], 
             'b-', label='DRN', linewidth=2, color=colors['drn'], alpha=0.7)
    ax2.plot(distortion_params, results['vit_pds_trend'], 
             'g-', label='ViT', linewidth=2, color=colors['vit'], alpha=0.7)
    ax2.plot(distortion_params, results['existing_gpass_pds_trend'], 
             'k--', label='Traditional GPass', linewidth=2, color=colors['traditional'], alpha=0.7)
    
    # Add annotation for hybrid model's superior diversity
    ax2.annotate('Superior Diversity', 
                xy=(0.4, results['hybrid_pds_trend'][2]), 
                xytext=(0.2, 0.7),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    ax2.set_xlabel('Distortion Parameter', fontsize=12)
    ax2.set_ylabel('PDS', fontsize=12)
    ax2.set_title('(b) Password Diversity Score (PDS)', fontsize=14, pad=15)
    ax2.grid(True, which='both', linestyle='-', alpha=0.2)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(0.2, 0.9)
    ax2.set_xlim(0.0, 0.8)
    
    # Plot 3: IRR Standard Deviation
    systems = ['DRN', 'ViT', 'Hybrid', 'Traditional']
    irr_stds = [
        results['drn_irr_std'],
        results['vit_irr_std'],
        results['hybrid_irr_std'],
        results['existing_gpass_irr_std']
    ]
    
    # Create bar plot with emphasis on hybrid model
    bars = ax3.bar(systems, irr_stds, color=[colors['drn'], colors['vit'], colors['hybrid'], colors['traditional']], width=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Highlight hybrid model's bar
    bars[2].set_edgecolor('black')
    bars[2].set_linewidth(2)
    
    ax3.set_ylabel('Standard Deviation', fontsize=12)
    ax3.set_title('(c) Standard Deviation for IRR', fontsize=14, pad=15)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, linestyle='-', alpha=0.2, axis='y')
    ax3.set_ylim(0, 0.18)
    
    # Plot 4: PDS Standard Deviation
    pds_stds = [
        results['drn_pds_std'],
        results['vit_pds_std'],
        results['hybrid_pds_std'],
        results['existing_gpass_pds_std']
    ]
    
    # Create bar plot with emphasis on hybrid model
    bars = ax4.bar(systems, pds_stds, color=[colors['drn'], colors['vit'], colors['hybrid'], colors['traditional']], width=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Highlight hybrid model's bar
    bars[2].set_edgecolor('black')
    bars[2].set_linewidth(2)
    
    ax4.set_ylabel('Standard Deviation', fontsize=12)
    ax4.set_title('(d) Standard Deviation for PDS', fontsize=14, pad=15)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, linestyle='-', alpha=0.2, axis='y')
    ax4.set_ylim(0, 0.10)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save the figure with high resolution
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    show_graphs() 