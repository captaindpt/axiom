import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML, display
import matplotlib
import os
matplotlib.use('Agg')  # Use non-interactive backend
matplotlib.rcParams['animation.embed_limit'] = 2**128

class BeliefSystemAnimator:
    """
    Real-time visualization and animation of belief system states and attacks.
    Provides dynamic visualization of:
    1. Pattern strength evolution
    2. Attack wave propagation
    3. System stability changes
    4. Belief network dynamics
    """
    
    def __init__(self, belief_system, figsize=(15, 10)):
        self.belief_system = belief_system
        self.figsize = figsize
        self.history = []
        self.attack_waves = []
        self.pattern_lines = {}
        self.tracked_patterns = set()
        
        # Setup custom colormaps
        self.pattern_cmap = plt.cm.Blues  # Use standard Blues colormap
        self.attack_cmap = self._create_attack_cmap()
        
        # Initialize the figure
        self.frame_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs('output/frames', exist_ok=True)
        
    def _create_attack_cmap(self):
        """Create custom colormap for attack visualization"""
        # Red for contradictions, white for neutral, green for reinforcement
        colors = ['#d73027', '#ffffff', '#1a9850']
        nodes = [0.0, 0.5, 1.0]
        return LinearSegmentedColormap.from_list('attacks', list(zip(nodes, colors)))
    
    def update_state(self, belief_matrix, attack_matrix=None):
        """
        Update visualization with current belief system state.
        
        Args:
            belief_matrix: Current belief matrix state
            attack_matrix: Optional attack matrix being applied
        """
        self.frame_count += 1
        
        # Create figure with two subplots if attack matrix is provided
        if attack_matrix is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot belief matrix
            sns.heatmap(belief_matrix, annot=True, cmap='viridis', 
                       vmin=0, vmax=1, ax=ax1, fmt='.2f')
            ax1.set_title('Current Belief State')
            
            # Plot attack matrix
            sns.heatmap(attack_matrix, annot=True, cmap='coolwarm',
                       vmin=-1, vmax=1, ax=ax2, fmt='.2f')
            ax2.set_title('Current Attack')
        else:
            fig, ax1 = plt.subplots(figsize=(8, 6))
            sns.heatmap(belief_matrix, annot=True, cmap='viridis',
                       vmin=0, vmax=1, ax=ax1, fmt='.2f')
            ax1.set_title('Current Belief State')
        
        plt.tight_layout()
        
        # Save the frame
        plt.savefig(f'output/frames/frame_{self.frame_count:03d}.png')
        plt.close()
        
        print(f"Frame saved: output/frames/frame_{self.frame_count:03d}.png")
    
    def plot_final_state(self):
        """Plot final state of the belief system"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot final belief matrix
        sns.heatmap(self.belief_system.P, annot=True, cmap='viridis',
                   vmin=0, vmax=1, ax=ax, fmt='.2f')
        ax.set_title('Final Belief System State')
        
        plt.tight_layout()
        
        # Save final state
        plt.savefig('output/frames/final_state.png')
        plt.close()
        
        print("Final state saved: output/frames/final_state.png")
    
    def show_current_state(self):
        """Show the current state"""
        fig, (ax_belief, ax_attack, ax_stability, ax_patterns) = self.setup_subplots()
        frame = len(self.history) - 1
        
        # Update belief matrix
        ax_belief.imshow(
            self.history[frame]['belief_matrix'],
            cmap=self.pattern_cmap,
            vmin=0, vmax=1,
            aspect='equal'
        )
        
        # Update attack wave
        ax_attack.imshow(
            self.attack_waves[frame],
            cmap=self.attack_cmap,
            vmin=-1, vmax=1,
            aspect='equal'
        )
        
        # Update stability timeline
        stabilities = [state['stability'] for state in self.history[:frame+1]]
        ax_stability.plot(range(len(stabilities)), stabilities, 'b-', linewidth=2)
        ax_stability.set_xlim(-1, max(20, len(stabilities)))
        
        # Update pattern evolution
        patterns_to_track = self._get_significant_patterns(frame)
        colors = plt.cm.tab10(np.linspace(0, 1, len(patterns_to_track)))
        
        for (pattern, coords), color in zip(patterns_to_track.items(), colors):
            strengths = [
                self.history[i]['belief_matrix'][coords]
                for i in range(frame + 1)
            ]
            ax_patterns.plot(
                range(len(strengths)), 
                strengths,
                label=pattern,
                linewidth=2,
                marker='o',
                markersize=4,
                color=color
            )
        
        ax_patterns.set_xlim(-1, max(20, frame))
        if patterns_to_track:
            ax_patterns.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add axis labels and ticks
        for ax in [ax_belief, ax_attack]:
            ax.set_xticks(range(self.belief_system.P.shape[0]))
            ax.set_yticks(range(self.belief_system.P.shape[1]))
        
        plt.show()
    
    def _get_significant_patterns(self, frame):
        """Identify significant patterns to track"""
        matrix = self.history[frame]['belief_matrix']
        significant = {}
        
        # Find top 5 strongest patterns
        flat_indices = np.argsort(matrix.flat)[-5:]
        for idx in flat_indices:
            i, j = np.unravel_index(idx, matrix.shape)
            if matrix[i, j] > 0.1:  # Only track if strength > 0.1
                significant[f"{i}→{j}"] = (i, j)
        
        return significant
    
    def animate_attack(self, attack_name):
        """Show the current state of the attack"""
        self.show_current_state()
    
    def setup_subplots(self):
        """Initialize the subplot layout"""
        plt.close('all')  # Close any existing figures
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
        
        # Belief Matrix Heatmap
        ax_belief = fig.add_subplot(gs[0, 0])
        belief_img = ax_belief.imshow(
            self.belief_system.P,
            cmap=self.pattern_cmap,
            vmin=0, vmax=1,
            aspect='equal'
        )
        ax_belief.set_title('Belief Matrix State')
        plt.colorbar(belief_img, ax=ax_belief, label='Pattern Strength')
        
        # Attack Wave Visualization
        ax_attack = fig.add_subplot(gs[0, 1])
        attack_img = ax_attack.imshow(
            np.zeros_like(self.belief_system.P),
            cmap=self.attack_cmap,
            vmin=-1, vmax=1,
            aspect='equal'
        )
        ax_attack.set_title('Attack Wave Propagation')
        plt.colorbar(attack_img, ax=ax_attack, label='Attack Impact')
        
        # Stability Timeline
        ax_stability = fig.add_subplot(gs[1, 0])
        ax_stability.set_title('System Stability')
        ax_stability.set_ylim(-0.1, 1.1)
        ax_stability.set_xlabel('Time Step')
        ax_stability.set_ylabel('Stability')
        ax_stability.grid(True)
        
        # Pattern Strength Evolution
        ax_patterns = fig.add_subplot(gs[1, 1])
        ax_patterns.set_title('Top Pattern Evolution')
        ax_patterns.set_ylim(-0.1, 1.1)
        ax_patterns.set_xlabel('Time Step')
        ax_patterns.set_ylabel('Pattern Strength')
        ax_patterns.grid(True)
        
        plt.tight_layout()
        return fig, (ax_belief, ax_attack, ax_stability, ax_patterns) 