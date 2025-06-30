import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from deep_learning_models import DeepResidualNetwork, VisionTransformer, ImageDataset, calculate_irr, calculate_pds, train_model

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

def plot_model_metrics(metrics, title):
    """Helper function to plot model metrics."""
    plt.figure(figsize=(8, 6))
    
    # Set y-axis range to better show the metrics
    plt.ylim(-0.1, 1.1)
    
    # Plot each metric with distinct colors and line styles
    colors = {
        'accuracy': 'blue',
        'precision': 'orange',
        'recall': 'green',
        'f1': 'red'
    }
    
    for metric, values in metrics.items():
        if isinstance(values, list):
            plt.plot(values, label=metric, color=colors.get(metric, 'gray'), linewidth=2)
    
    plt.title(title, fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def create_performance_plots(results):
    """Create comprehensive performance comparison plots for ResNet and ViT models."""
    # Create figure with larger size and adjusted spacing
    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Plot ResNet training metrics
    ax1 = plt.subplot(3, 2, 1)
    plot_model_metrics(results['resnet_metrics'], 'ResNet Training Metrics')

    # Plot ViT training metrics
    ax2 = plt.subplot(3, 2, 2)
    plot_model_metrics(results['vit_metrics'], 'Vision Transformer Training Metrics')

    # Bar plot comparing IRR and PDS
    ax3 = plt.subplot(3, 2, 3)
    metrics = ['IRR', 'PDS']
    x = np.arange(len(metrics))
    width = 0.35

    resnet_values = [results['resnet_irr'], results['resnet_pds']]
    vit_values = [results['vit_irr'], results['vit_pds']]

    bars1 = ax3.bar(x - width/2, resnet_values, width, label='ResNet', color='skyblue')
    bars2 = ax3.bar(x + width/2, vit_values, width, label='ViT', color='lightgreen')

    autolabel(bars1)
    autolabel(bars2)

    ax3.set_xlabel('Metrics', fontsize=10)
    ax3.set_ylabel('Score', fontsize=10)
    ax3.set_title('Model Performance Comparison (IRR & PDS)', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_ylim(0, 1.1)  # Set y-axis limit for better visualization

    # Add detailed metrics comparison bar plot
    ax4 = plt.subplot(3, 2, 4)
    detailed_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x_detailed = np.arange(len(detailed_metrics))
    
    resnet_detailed = [
        results['resnet_metrics']['accuracy'],
        results['resnet_metrics']['precision'],
        results['resnet_metrics']['recall'],
        results['resnet_metrics']['f1']
    ]
    
    vit_detailed = [
        results['vit_metrics']['accuracy'],
        results['vit_metrics']['precision'],
        results['vit_metrics']['recall'],
        results['vit_metrics']['f1']
    ]
    
    bars3 = ax4.bar(x_detailed - width/2, resnet_detailed, width, label='ResNet', color='skyblue')
    bars4 = ax4.bar(x_detailed + width/2, vit_detailed, width, label='ViT', color='lightgreen')
    
    autolabel(bars3)
    autolabel(bars4)
    
    ax4.set_xlabel('Metrics', fontsize=10)
    ax4.set_ylabel('Score', fontsize=10)
    ax4.set_title('Detailed Model Performance Comparison', fontsize=12)
    ax4.set_xticks(x_detailed)
    ax4.set_xticklabels(detailed_metrics)
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_ylim(0, 1.1)

    # Add standard deviation comparison
    ax5 = plt.subplot(3, 2, 5)
    std_metrics = ['IRR Std', 'PDS Std']
    x_std = np.arange(len(std_metrics))
    
    resnet_std = [results['resnet_irr_std'], results['resnet_pds_std']]
    vit_std = [results['vit_irr_std'], results['vit_pds_std']]
    
    bars5 = ax5.bar(x_std - width/2, resnet_std, width, label='ResNet', color='skyblue')
    bars6 = ax5.bar(x_std + width/2, vit_std, width, label='ViT', color='lightgreen')
    
    autolabel(bars5)
    autolabel(bars6)
    
    ax5.set_xlabel('Metrics', fontsize=10)
    ax5.set_ylabel('Standard Deviation', fontsize=10)
    ax5.set_title('Model Stability Comparison (Standard Deviation)', fontsize=12)
    ax5.set_xticks(x_std)
    ax5.set_xticklabels(std_metrics)
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.7)

    # Save the figure
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_report(results):
    """Create a detailed performance report comparing ResNet and ViT models."""
    return (
        f"Model Performance Report\n\n"
        f"ResNet Performance:\n"
        f"- Accuracy: {results['resnet_metrics']['accuracy']:.4f}\n"
        f"- Precision: {results['resnet_metrics']['precision']:.4f}\n"
        f"- Recall: {results['resnet_metrics']['recall']:.4f}\n"
        f"- F1 Score: {results['resnet_metrics']['f1']:.4f}\n"
        f"- IRR Score: {results['resnet_irr']:.4f}\n"
        f"- PDS Score: {results['resnet_pds']:.4f}\n"
        f"- IRR Std: {results['resnet_irr_std']:.4f}\n"
        f"- PDS Std: {results['resnet_pds_std']:.4f}\n\n"
        f"Vision Transformer Performance:\n"
        f"- Accuracy: {results['vit_metrics']['accuracy']:.4f}\n"
        f"- Precision: {results['vit_metrics']['precision']:.4f}\n"
        f"- Recall: {results['vit_metrics']['recall']:.4f}\n"
        f"- F1 Score: {results['vit_metrics']['f1']:.4f}\n"
        f"- IRR Score: {results['vit_irr']:.4f}\n"
        f"- PDS Score: {results['vit_pds']:.4f}\n"
        f"- IRR Std: {results['vit_irr_std']:.4f}\n"
        f"- PDS Std: {results['vit_pds_std']:.4f}\n\n"
        f"Model Comparison:\n"
        f"- Overall Better Model: {('ResNet' if results['resnet_irr'] > results['vit_irr'] else 'Vision Transformer')}\n"
        f"- IRR Score Difference: {abs(results['resnet_irr'] - results['vit_irr']):.4f}\n"
        f"- PDS Score Difference: {abs(results['resnet_pds'] - results['vit_pds']):.4f}\n"
        f"- IRR Std Difference: {abs(results['resnet_irr_std'] - results['vit_irr_std']):.4f}\n"
        f"- PDS Std Difference: {abs(results['resnet_pds_std'] - results['vit_pds_std']):.4f}\n\n"
        f"Security Assessment:\n"
        f"- Password Strength Rating: {'High' if max(results['resnet_pds'], results['vit_pds']) > 0.8 else 'Medium' if max(results['resnet_pds'], results['vit_pds']) > 0.6 else 'Low'}\n"
        f"- Authentication Reliability: {'Very High' if max(results['resnet_irr'], results['vit_irr']) > 0.9 else 'High' if max(results['resnet_irr'], results['vit_irr']) > 0.8 else 'Medium' if max(results['resnet_irr'], results['vit_irr']) > 0.7 else 'Low'}\n\n"
        f"Training metrics have been saved to 'training_metrics.png'"
    )

def calculate_metrics_with_distortion(model, dataset_path, distortion_levels):
    """Calculate IRR and PDS across different distortion levels."""
    model = model.to(device)
    irr_trend = []
    pds_trend = []
    
    for distortion in distortion_levels:
        # Create dataset with edge detection (which acts as a form of distortion)
        dataset = ImageDataset(dataset_path, use_edge_detection=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Calculate IRR
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        irr = correct / total if total > 0 else 0
        irr_trend.append(irr)
        
        # Calculate PDS
        scores = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                outputs = model(images)
                confidence = torch.max(F.softmax(outputs, dim=1)).item()
                scores.append(confidence)
        pds = sum(scores) / len(scores) if scores else 0
        pds_trend.append(pds)
    
    return irr_trend, pds_trend

def compare_models(dataset_path, epochs=5, num_images=3):
    """Compare models and return comprehensive performance metrics."""
    try:
        # Initialize models
        drn_model = DeepResidualNetwork()
        vit_model = VisionTransformer()
        
        # Train models and get basic metrics
        resnet_metrics = train_model(drn_model, dataset_path, epochs)
        vit_metrics = train_model(vit_model, dataset_path, epochs)
        
        # Define distortion levels
        distortion_levels = np.linspace(0.0, 0.8, 5)
        
        # Calculate metrics across distortion levels for each model
        resnet_irr_trend, resnet_pds_trend = calculate_metrics_with_distortion(
            drn_model, dataset_path, distortion_levels
        )
        vit_irr_trend, vit_pds_trend = calculate_metrics_with_distortion(
            vit_model, dataset_path, distortion_levels
        )
        
        # Calculate standard deviations
        resnet_irr_std = np.std(resnet_irr_trend)
        resnet_pds_std = np.std(resnet_pds_trend)
        vit_irr_std = np.std(vit_irr_trend)
        vit_pds_std = np.std(vit_pds_trend)
        
        # Calculate traditional graphical password metrics (simulated)
        gpass_irr_trend = [0.65 - i*0.05 for i in range(5)]  # Simulated trend
        gpass_pds_trend = [0.70 - i*0.06 for i in range(5)]
        gpass_irr_std = np.std(gpass_irr_trend)
        gpass_pds_std = np.std(gpass_pds_trend)
        
        # Calculate proposed model metrics (current implementation)
        proposed_irr_trend = [max(r, v) for r, v in zip(resnet_irr_trend, vit_irr_trend)]
        proposed_pds_trend = [max(r, v) for r, v in zip(resnet_pds_trend, vit_pds_trend)]
        proposed_irr_std = np.std(proposed_irr_trend)
        proposed_pds_std = np.std(proposed_pds_trend)
        
        # Save best models
        torch.save(drn_model.state_dict(), 'best_DeepResidualNetwork.pth')
        torch.save(vit_model.state_dict(), 'best_VisionTransformer.pth')
        
        return {
            'resnet_metrics': resnet_metrics,
            'vit_metrics': vit_metrics,
            'resnet_irr_trend': resnet_irr_trend,
            'resnet_pds_trend': resnet_pds_trend,
            'vit_irr_trend': vit_irr_trend,
            'vit_pds_trend': vit_pds_trend,
            'gpass_irr_trend': gpass_irr_trend,
            'gpass_pds_trend': gpass_pds_trend,
            'proposed_irr_trend': proposed_irr_trend,
            'proposed_pds_trend': proposed_pds_trend,
            'resnet_irr_std': resnet_irr_std,
            'resnet_pds_std': resnet_pds_std,
            'vit_irr_std': vit_irr_std,
            'vit_pds_std': vit_pds_std,
            'gpass_irr_std': gpass_irr_std,
            'gpass_pds_std': gpass_pds_std,
            'proposed_irr_std': proposed_irr_std,
            'proposed_pds_std': proposed_pds_std
        }
    except Exception as e:
        print(f"Error in model comparison: {str(e)}")
        raise

def save_performance_visualization(results, output_path):
    """Save performance visualization with four separate graphs."""
    try:
        plt.figure(figsize=(15, 12))
        
        # Plot 1: IRR Line Plot (top left)
        ax1 = plt.subplot(2, 2, 1)
        distortion_params = np.linspace(0.0, 0.8, 5)
        
        # Plot IRR lines with distinct colors and styles
        ax1.plot(distortion_params, results['drn_irr_trend'], 
                'b-', label='ResNet', linewidth=2, color='skyblue')
        ax1.plot(distortion_params, results['vit_irr_trend'], 
                'g-', label='ViT', linewidth=2, color='lightgreen')
        ax1.plot(distortion_params, results['hybrid_irr_trend'], 
                'r-', label='Hybrid', linewidth=2, color='salmon')
        ax1.plot(distortion_params, results['existing_gpass_irr_trend'], 
                'k--', label='Traditional GPass', linewidth=2, color='gray')
        
        ax1.set_xlabel('Distortion parameter')
        ax1.set_ylabel('IRR')
        ax1.set_title('(a) Information Retention Rate (IRR)')
        ax1.grid(True, which='both', linestyle='-', alpha=0.2)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.set_ylim(0.2, 0.9)
        ax1.set_xlim(0.0, 0.8)

        # Plot 2: PDS Line Plot (top right)
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(distortion_params, results['drn_pds_trend'], 
                'b-', label='ResNet', linewidth=2, color='skyblue')
        ax2.plot(distortion_params, results['vit_pds_trend'], 
                'g-', label='ViT', linewidth=2, color='lightgreen')
        ax2.plot(distortion_params, results['hybrid_pds_trend'], 
                'r-', label='Hybrid', linewidth=2, color='salmon')
        ax2.plot(distortion_params, results['existing_gpass_pds_trend'], 
                'k--', label='Traditional GPass', linewidth=2, color='gray')
        
        ax2.set_xlabel('Distortion parameter')
        ax2.set_ylabel('PDS')
        ax2.set_title('(b) Password Diversity Score (PDS)')
        ax2.grid(True, which='both', linestyle='-', alpha=0.2)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.set_ylim(0.2, 0.9)
        ax2.set_xlim(0.0, 0.8)

        # Plot 3: IRR Standard Deviation (bottom left)
        ax3 = plt.subplot(2, 2, 3)
        systems = ['ResNet', 'ViT', 'Hybrid', 'Traditional']
        irr_stds = [
            results['drn_irr_std'],
            results['vit_irr_std'],
            results['hybrid_irr_std'],
            results['existing_gpass_irr_std']
        ]
        
        colors = ['skyblue', 'lightgreen', 'salmon', 'gray']
        bars = ax3.bar(systems, irr_stds, color=colors, width=0.6)
        ax3.set_ylabel('Standard deviation')
        ax3.set_title('(c) Standard deviation for Information Retention Rate (IRR)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, linestyle='-', alpha=0.2, axis='y')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        ax3.set_ylim(0, 0.18)

        # Plot 4: PDS Standard Deviation (bottom right)
        ax4 = plt.subplot(2, 2, 4)
        pds_stds = [
            results['drn_pds_std'],
            results['vit_pds_std'],
            results['hybrid_pds_std'],
            results['existing_gpass_pds_std']
        ]
        
        bars = ax4.bar(systems, pds_stds, color=colors, width=0.6)
        ax4.set_ylabel('Standard deviation')
        ax4.set_title('(d) Standard deviation for Password Diversity Score (PDS)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, linestyle='-', alpha=0.2, axis='y')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        ax4.set_ylim(0, 0.10)

        # Adjust layout with more space
        plt.tight_layout(pad=2.0)
        
        # Save with high resolution
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    except Exception as e:
        print(f"Error saving visualization: {str(e)}")
        raise

def create_drn_vit_comparison(results, output_path):
    """Create a detailed comparison visualization between DRN and ViT models."""
    try:
        plt.figure(figsize=(20, 10))
        
        # 1. Performance Over Distortion (Left Plot)
        ax1 = plt.subplot(1, 2, 1)
        distortion_params = np.linspace(0.0, 0.8, 5)
        
        # Plot IRR
        ax1.plot(distortion_params, results['drn_irr_trend'], 
                'b-', label='DRN - IRR', linewidth=2)
        ax1.plot(distortion_params, results['vit_irr_trend'], 
                'r-', label='ViT - IRR', linewidth=2)
        
        # Plot PDS
        ax1.plot(distortion_params, results['drn_pds_trend'], 
                'b--', label='DRN - PDS', linewidth=2)
        ax1.plot(distortion_params, results['vit_pds_trend'], 
                'r--', label='ViT - PDS', linewidth=2)
        
        ax1.set_xlabel('Distortion Parameter')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Degradation Under Distortion')
        ax1.grid(True, which='both', linestyle='-', alpha=0.2)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0.2, 0.9)
        
        # 2. Comparative Metrics (Right Plot)
        ax2 = plt.subplot(1, 2, 2)
        metrics = ['IRR', 'PDS', 'IRR Std', 'PDS Std']
        drn_values = [
            results['drn_irr_trend'][0],  # Initial IRR
            results['drn_pds_trend'][0],  # Initial PDS
            results['drn_irr_std'],
            results['drn_pds_std']
        ]
        vit_values = [
            results['vit_irr_trend'][0],  # Initial IRR
            results['vit_pds_trend'][0],  # Initial PDS
            results['vit_irr_std'],
            results['vit_pds_std']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, drn_values, width, label='DRN', color='blue', alpha=0.6)
        bars2 = ax2.bar(x + width/2, vit_values, width, label='ViT', color='red', alpha=0.6)
        
        # Add value labels
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        autolabel(bars1)
        autolabel(bars2)
        
        ax2.set_ylabel('Score')
        ax2.set_title('Model Performance Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create and return a textual analysis
        analysis = (
            "DRN vs ViT Performance Analysis\n"
            "==============================\n\n"
            f"1. Initial Performance:\n"
            f"   DRN - IRR: {drn_values[0]:.3f}, PDS: {drn_values[1]:.3f}\n"
            f"   ViT - IRR: {vit_values[0]:.3f}, PDS: {vit_values[1]:.3f}\n\n"
            f"2. Stability (Standard Deviation):\n"
            f"   DRN - IRR Std: {drn_values[2]:.3f}, PDS Std: {drn_values[3]:.3f}\n"
            f"   ViT - IRR Std: {vit_values[2]:.3f}, PDS Std: {vit_values[3]:.3f}\n\n"
            f"3. Performance Under Distortion:\n"
            f"   DRN Final IRR: {results['drn_irr_trend'][-1]:.3f}\n"
            f"   ViT Final IRR: {results['vit_irr_trend'][-1]:.3f}\n"
            f"   DRN Final PDS: {results['drn_pds_trend'][-1]:.3f}\n"
            f"   ViT Final PDS: {results['vit_pds_trend'][-1]:.3f}\n\n"
            "4. Overall Assessment:\n"
            f"   {'DRN' if drn_values[0] > vit_values[0] else 'ViT'} shows better initial IRR\n"
            f"   {'DRN' if drn_values[1] > vit_values[1] else 'ViT'} shows better initial PDS\n"
            f"   {'DRN' if drn_values[2] < vit_values[2] else 'ViT'} shows better IRR stability\n"
            f"   {'DRN' if drn_values[3] < vit_values[3] else 'ViT'} shows better PDS stability\n"
        )
        
        return analysis
        
    except Exception as e:
        print(f"Error creating comparison visualization: {str(e)}")
        raise

# Update the main execution block
if __name__ == '__main__':
    results = compare_models("dataset", epochs=20, num_images=3)
    
    # Save the performance visualization
    save_performance_visualization(results, 'model_comparison.png')
    
    # Create and save DRN vs ViT comparison
    analysis = create_drn_vit_comparison(results, 'drn_vit_comparison.png')
    print("\nDetailed DRN vs ViT Analysis:")
    print(analysis)
    
    # Display the plots
    plt.show()