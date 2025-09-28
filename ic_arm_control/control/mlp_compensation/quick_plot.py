#!/usr/bin/env python3
"""
Quick visualization script for MLP model performance
"""

import numpy as np
import matplotlib.pyplot as plt
from mlp_gravity_compensation import LightweightMLPGravityCompensation


def quick_performance_plot():
    """Generate quick performance visualization"""
    # Load model
    mlp_system = LightweightMLPGravityCompensation()
    mlp_system.load_model('mlp_gravity_model.pkl')

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. R² scores per joint
    joints = range(1, 7)
    train_scores = mlp_system.train_scores
    val_scores = mlp_system.val_scores

    x = np.arange(len(joints))
    width = 0.35

    ax1.bar(x - width/2, train_scores, width, label='Train R²', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, val_scores, width, label='Validation R²', alpha=0.8, color='lightcoral')
    ax1.set_xlabel('Joint')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Scores per Joint')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'J{i}' for i in joints])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.2, 1.0)

    # 2. Performance radar chart
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    val_scores_complete = val_scores + [val_scores[0]]

    ax2 = plt.subplot(2, 2, 2, projection='polar')
    ax2.plot(angles, val_scores_complete, 'o-', linewidth=2, label='Validation R²')
    ax2.fill(angles, val_scores_complete, alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([f'J{i}' for i in joints])
    ax2.set_ylim(0, 1)
    ax2.set_title('Validation R² - Radar View')
    ax2.grid(True)

    # 3. Performance summary
    avg_train = np.mean(train_scores)
    avg_val = np.mean(val_scores)
    best_joint = np.argmax(val_scores) + 1
    worst_joint = np.argmin(val_scores) + 1

    summary_text = f"Model Performance Summary\n\n"
    summary_text += f"Average Train R²: {avg_train:.4f}\n"
    summary_text += f"Average Val R²: {avg_val:.4f}\n\n"
    summary_text += f"Best Joint: J{best_joint} (R² = {max(val_scores):.4f})\n"
    summary_text += f"Worst Joint: J{worst_joint} (R² = {min(val_scores):.4f})\n\n"

    if avg_val > 0.8:
        rating = "Excellent"
    elif avg_val > 0.6:
        rating = "Good"
    elif avg_val > 0.4:
        rating = "Fair"
    else:
        rating = "Poor"

    summary_text += f"Overall Rating: {rating}"

    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Performance Summary')

    # 4. Performance distribution
    all_scores = train_scores + val_scores
    labels = [f'J{i}_T' for i in joints] + [f'J{i}_V' for i in joints]
    colors = ['skyblue'] * 6 + ['lightcoral'] * 6

    bars = ax4.bar(labels, all_scores, color=colors)
    ax4.set_ylabel('R² Score')
    ax4.set_title('All Joint Performance')
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.2, 1.0)

    # Add value labels on bars
    for bar, score in zip(bars, all_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('mlp_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


def prediction_test_plot():
    """Test predictions and plot results"""
    mlp_system = LightweightMLPGravityCompensation()
    mlp_system.load_model('mlp_gravity_model.pkl')

    # Test positions
    test_positions = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.3, 0.2, -0.1, 0.4, -0.2],
        [1.0, -0.5, 0.5, -0.3, 0.8, -0.4],
        [0.2, 0.3, -0.1, 0.6, -0.2, 0.4],
        [-0.3, 0.7, 0.1, -0.5, 0.3, -0.6]
    ])

    # Get predictions
    predictions = []
    for pos in test_positions:
        pred = mlp_system.compute_gravity_compensation(pos)
        predictions.append(pred)

    predictions = np.array(predictions)

    # Create prediction heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Heatmap of predictions
    im = ax1.imshow(predictions.T, aspect='auto', cmap='RdBu_r', vmin=-5, vmax=5)
    ax1.set_xlabel('Test Case')
    ax1.set_ylabel('Joint')
    ax1.set_title('Predicted Gravity Torques (Nm)')
    ax1.set_yticks(range(6))
    ax1.set_yticklabels([f'J{i+1}' for i in range(6)])
    ax1.set_xticks(range(5))
    ax1.set_xticklabels([f'Test {i+1}' for i in range(5)])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Torque (Nm)')

    # Torque magnitude plot
    torque_magnitudes = np.abs(predictions)
    for i in range(6):
        ax2.plot(range(5), torque_magnitudes[:, i], 'o-', label=f'Joint {i+1}', linewidth=2, markersize=6)

    ax2.set_xlabel('Test Case')
    ax2.set_ylabel('Torque Magnitude (Nm)')
    ax2.set_title('Torque Magnitude by Joint')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels([f'Test {i+1}' for i in range(5)])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5.5)

    plt.tight_layout()
    plt.savefig('mlp_prediction_test.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main plotting function"""
    print("=== Quick MLP Performance Visualization ===")

    print("1. Generating performance summary plot...")
    quick_performance_plot()

    print("2. Generating prediction test plot...")
    prediction_test_plot()

    print("\n✅ Plots generated successfully!")
    print("Generated files:")
    print("- mlp_performance_summary.png")
    print("- mlp_prediction_test.png")


if __name__ == "__main__":
    main()