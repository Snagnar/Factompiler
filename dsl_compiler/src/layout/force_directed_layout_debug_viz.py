"""Debug visualization for force-directed layout optimization.

This module provides visualization capabilities to debug and understand
the force-directed layout algorithm. It can render the graph state at each
iteration and compile all frames into an animated GIF.

Usage:
    # In force_directed_layout.py, enable visualization:
    visualizer = LayoutVisualizer(layout_engine)
    callback = visualizer.create_callback()

    # Pass callback to scipy.optimize.minimize
    result = minimize(..., callback=callback)

    # After optimization, create GIF
    visualizer.create_gif("output.gif")
"""

import os
import shutil
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np

# Optional dependencies for visualization
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import LineCollection

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualization disabled")

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available, GIF creation disabled")

if TYPE_CHECKING:
    from .force_directed_layout import ForceDirectedLayoutEngine


class LayoutVisualizer:
    """Visualizes force-directed layout optimization progress."""

    def __init__(
        self,
        layout_engine: "ForceDirectedLayoutEngine",
        attempt_id: int,
        phase_name: str = "",
        output_base_dir: Optional[str] = None,
        frame_skip: int = 1,
        figsize: tuple = (12, 12),
        dpi: int = 100,
    ):
        """Initialize visualizer.

        Args:
            layout_engine: The layout engine to visualize
            attempt_id: Unique ID for this optimization attempt
            phase_name: Name of optimization phase (e.g., "phase1", "phase2")
            output_base_dir: Base directory to save GIFs (default: output/)
            frame_skip: Only save every Nth frame (1 = all frames)
            figsize: Figure size in inches
            dpi: Resolution for saved images
        """
        if not HAS_MATPLOTLIB:
            raise RuntimeError("matplotlib is required for visualization")

        self.engine = layout_engine
        self.attempt_id = attempt_id
        self.phase_name = phase_name
        self.frame_skip = frame_skip
        self.figsize = figsize
        self.dpi = dpi

        # Setup output directories
        if output_base_dir is None:
            self.base_dir = Path("output/layout_visualization")
        else:
            self.base_dir = Path(output_base_dir)

        # Create unique temp directory for this attempt (not per phase)
        self.temp_dir = self.base_dir / f"temp_attempt_{attempt_id}"

        # Clear and create temp directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.frame_count = 0
        self.iteration_count = 0

        # Track current phase and energy (can be updated during optimization)
        self.current_phase = phase_name
        self.current_energy = 0.0

        print(f"  Visualization: capturing frames to {self.temp_dir}")

    def set_phase(self, phase_name: str):
        """Set the current optimization phase.

        Args:
            phase_name: Name of the phase (e.g., "Phase 1", "Phase 2")
        """
        self.current_phase = phase_name
        print(f"  Visualization: entering {phase_name}")

    def create_callback(self, energy_func=None):
        """Create a callback function for scipy.optimize.minimize.

        Args:
            energy_func: Optional function to compute energy from position vector

        Returns:
            Callback function that can be passed to minimize()
        """

        def callback(xk):
            """Callback called after each optimization iteration.

            Args:
                xk: Current position vector (flattened)
            """
            self.iteration_count += 1

            # Compute energy if function provided
            if energy_func is not None:
                result = energy_func(xk)
                # Handle both scalar and (energy, gradient) tuple returns
                if isinstance(result, tuple):
                    self.current_energy = result[0]
                else:
                    self.current_energy = result

            # Only save every Nth frame
            if self.iteration_count % self.frame_skip == 0:
                self.visualize_iteration(xk, self.iteration_count)

        return callback

    def visualize_iteration(self, pos_flat: np.ndarray, iteration: int):
        """Visualize current state of the layout.

        Args:
            pos_flat: Flattened position array
            iteration: Current iteration number
        """
        # Reshape positions
        pos_array = pos_flat.reshape(-1, 2)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Draw connections first (so they appear behind entities)
        self._draw_connections(ax, pos_array)

        # Draw entities
        self._draw_entities(ax, pos_array)

        # Set axis properties
        self._setup_axes(ax, pos_array, iteration)

        # Save frame
        frame_path = self.temp_dir / f"frame_{self.frame_count:04d}.png"
        plt.savefig(frame_path, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)

        self.frame_count += 1

        if self.frame_count % 10 == 0:
            print(f"    Captured {self.frame_count} frames...")

    def _draw_connections(self, ax, pos_array: np.ndarray):
        """Draw connection lines between entities.

        Args:
            ax: Matplotlib axes
            pos_array: Position array (n_entities, 2)
        """
        if len(self.engine.connection_pairs) == 0:
            return

        # Get all connection line segments
        segments = []
        for idx1, idx2 in self.engine.connection_pairs:
            pos1 = pos_array[idx1]
            pos2 = pos_array[idx2]
            segments.append([pos1, pos2])

        # Draw all connections as a LineCollection (efficient)
        lc = LineCollection(
            segments,
            colors="blue",
            alpha=0.3,
            linewidths=1.0,
            zorder=1,  # Behind entities
        )
        ax.add_collection(lc)

    def _draw_entities(self, ax, pos_array: np.ndarray):
        """Draw entity rectangles.

        Args:
            ax: Matplotlib axes
            pos_array: Position array (n_entities, 2)
        """
        for i, entity_id in enumerate(self.engine.entity_ids):
            x, y = pos_array[i]
            width, height = self.engine.footprints[i]

            # Determine color based on whether entity is fixed
            is_fixed = i in self.engine.fixed_indices
            color = "red" if is_fixed else "lightblue"
            edge_color = "darkred" if is_fixed else "darkblue"

            # Create rectangle (centered at x, y)
            rect = Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=1.5,
                alpha=0.7,
                zorder=2,  # In front of connections
            )
            ax.add_patch(rect)

            # Add entity label
            ax.text(
                x,
                y,
                entity_id[:8],  # Truncate long IDs
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                zorder=3,
            )

    def _setup_axes(self, ax, pos_array: np.ndarray, iteration: int):
        """Setup axis limits, labels, and title.

        Args:
            ax: Matplotlib axes
            pos_array: Position array (n_entities, 2)
            iteration: Current iteration number
        """
        # Calculate bounds with padding
        if len(pos_array) > 0:
            # Account for entity sizes in bounds
            all_xs = []
            all_ys = []
            for i in range(len(pos_array)):
                x, y = pos_array[i]
                w, h = self.engine.footprints[i]
                all_xs.extend([x - w / 2, x + w / 2])
                all_ys.extend([y - h / 2, y + h / 2])

            x_min, x_max = min(all_xs), max(all_xs)
            y_min, y_max = min(all_ys), max(all_ys)

            # Add padding
            padding = 5.0
            x_range = max(x_max - x_min, 10)
            y_range = max(y_max - y_min, 10)

            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)
        else:
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Labels and title
        ax.set_xlabel("X Position", fontsize=10)
        ax.set_ylabel("Y Position", fontsize=10)

        # Build title with phase and energy information
        title_parts = ["Force-Directed Layout"]
        if self.current_phase:
            title_parts.append(f"- {self.current_phase}")
        title_parts.append(f"- Iteration {iteration}")

        # Build subtitle with stats
        subtitle_parts = [f"Attempt {self.attempt_id}"]
        if self.current_energy > 0:
            subtitle_parts.append(f"Energy: {self.current_energy:.2f}")
        subtitle_parts.append(
            f"{len(self.engine.entity_ids)} entities, "
            f"{len(self.engine.connection_pairs)} connections"
        )

        ax.set_title(
            " ".join(title_parts) + "\n" + " | ".join(subtitle_parts),
            fontsize=12,
            fontweight="bold",
        )

        # Grid
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    def create_gif(
        self,
        duration: int = 100,
        loop: int = 0,
    ) -> Optional[str]:
        """Compile all saved frames into an animated GIF.

        Args:
            duration: Duration of each frame in milliseconds
            loop: Number of loops (0 = infinite)

        Returns:
            Path to the created GIF, or None if failed
        """
        if not HAS_PIL:
            print("  Error: PIL is required to create GIFs")
            return None

        # Get all frame files sorted
        frame_files = sorted(self.temp_dir.glob("frame_*.png"))

        if not frame_files:
            print(f"  No frames found in {self.temp_dir}")
            return None

        # Generate output filename (no phase suffix - phases shown in GIF title)
        gif_name = f"attempt_{self.attempt_id}.gif"

        output_path = self.base_dir / gif_name

        print(f"  Creating GIF from {len(frame_files)} frames...")

        # Load all frames
        frames = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            frames.append(img)

        # Save as GIF
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=loop,
                optimize=False,  # Don't optimize to preserve quality
            )

            # Print file size
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✓ GIF saved: {output_path} ({file_size_mb:.2f} MB)")

            return str(output_path)
        else:
            print("  No frames to create GIF")
            return None

    def cleanup(self):
        """Remove temporary frame files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"  Cleaned up temporary frames from: {self.temp_dir}")


def visualize_graph(
    layout_engine: "ForceDirectedLayoutEngine",
    pos_flat: np.ndarray,
    output_path: str = "layout_snapshot.png",
    figsize: tuple = (12, 12),
    dpi: int = 150,
):
    """Create a one-off visualization of the current layout state.

    This is a convenience function for visualizing a single state
    without setting up the full animation workflow.

    Args:
        layout_engine: The layout engine
        pos_flat: Flattened position array
        output_path: Where to save the image
        figsize: Figure size in inches
        dpi: Resolution
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization")
        return

    pos_array = pos_flat.reshape(-1, 2)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Draw connections
    if len(layout_engine.connection_pairs) > 0:
        segments = []
        for idx1, idx2 in layout_engine.connection_pairs:
            pos1 = pos_array[idx1]
            pos2 = pos_array[idx2]
            segments.append([pos1, pos2])

        lc = LineCollection(segments, colors="blue", alpha=0.3, linewidths=1.5)
        ax.add_collection(lc)

    # Draw entities
    for i, entity_id in enumerate(layout_engine.entity_ids):
        x, y = pos_array[i]
        width, height = layout_engine.footprints[i]

        is_fixed = i in layout_engine.fixed_indices
        color = "red" if is_fixed else "lightblue"
        edge_color = "darkred" if is_fixed else "darkblue"

        rect = Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            alpha=0.7,
        )
        ax.add_patch(rect)

        ax.text(x, y, entity_id[:10], ha="center", va="center", fontsize=7)

    # Setup axes
    if len(pos_array) > 0:
        all_xs = []
        all_ys = []
        for i in range(len(pos_array)):
            x, y = pos_array[i]
            w, h = layout_engine.footprints[i]
            all_xs.extend([x - w / 2, x + w / 2])
            all_ys.extend([y - h / 2, y + h / 2])

        padding = 5.0
        ax.set_xlim(min(all_xs) - padding, max(all_xs) + padding)
        ax.set_ylim(min(all_ys) - padding, max(all_ys) + padding)

    ax.set_aspect("equal")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(
        f"Force-Directed Layout\n"
        f"{len(layout_engine.entity_ids)} entities, "
        f"{len(layout_engine.connection_pairs)} connections"
    )
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    print(f"Visualization saved to: {output_path}")
