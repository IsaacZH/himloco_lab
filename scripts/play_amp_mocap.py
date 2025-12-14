"""
AMP Motion Replayer for Go2 Robot

This script replays AMP motion capture data for the Go2 robot in Isaac Lab.
It loads all motion files from the mocap_motions_go2 dataset directory.

Usage:
    python scripts/play_amp_mocap.py [options]

Options:
    --device DEVICE        Device to run simulation on (default: cuda:0)
    --loop                 Enable continuous looping of the motion (default: True)

Example:
    # Replay motion with default settings
    python scripts/play_amp_mocap.py

    # Use a different device
    python scripts/play_amp_mocap.py --device cpu
"""

import sys
import os
import argparse
import time
import glob
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from isaaclab.app import AppLauncher

# Motion files directory
MOTION_DIR = "source/himloco_lab/himloco_lab/datasets/mocap_motions_go2"

# Configuration: whether to fix robot rotation (keep it unchanged)
FIX_BASE_ROTATION = True  # Set to False to follow AMP rotation data

# Command line arguments
parser = argparse.ArgumentParser(description="Play AMP mocap data in Isaac Lab")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--loop", action="store_true", default=True, help="Loop the motion continuously")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.ui

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation
from isaaclab.ui.widgets import LiveLinePlot

# Local imports
from himloco_lab.rsl_rl.datasets.motion_loader import AMPLoader
from himloco_lab.assets.unitree import UNITREE_GO2_CFG


class RealTimePlotter:
    """Real-time plotter for robot state data using Isaac Lab's LiveLinePlot widget."""
    
    def __init__(self, max_points=500):
        """Initialize the plotter.
        
        Args:
            max_points: Maximum number of points to display in the plot
        """
        self.max_points = max_points
        
        # Create UI window and plots inside it
        self.window = omni.ui.Window("Robot State Monitor", width=400, height=800)
        
        with self.window.frame:
            with omni.ui.VStack(spacing=10):
                # Status information section
                with omni.ui.CollapsableFrame("Playback Status", collapsed=False):
                    with omni.ui.VStack(spacing=5):
                        self.motion_label = omni.ui.Label(
                            "Motion: Initializing...",
                            style={"font_size": 14, "color": 0xFF00FF00}
                        )
                        self.frame_label = omni.ui.Label(
                            "Frame: 0 / 0",
                            style={"font_size": 12, "color": 0xFFFFFFFF}
                        )
                
                # Position plot
                with omni.ui.CollapsableFrame("Position (m)", collapsed=False):
                    self.pos_plot = LiveLinePlot(
                        y_data=[[], [], []],
                        y_min=-2.0,
                        y_max=2.0,
                        plot_height=150,
                        show_legend=True,
                        legends=["X (m)", "Y (m)", "Z (m)"],
                        max_datapoints=max_points,
                    )
                
                # Rotation plot
                with omni.ui.CollapsableFrame("Rotation (quat)", collapsed=False):
                    self.rot_plot = LiveLinePlot(
                        y_data=[[], [], [], []],
                        y_min=-1.0,
                        y_max=1.0,
                        plot_height=150,
                        show_legend=True,
                        legends=["W", "X", "Y", "Z"],
                        max_datapoints=max_points,
                    )
                
                # Linear velocity plot
                with omni.ui.CollapsableFrame("Linear Velocity (m/s)", collapsed=False):
                    self.lin_vel_plot = LiveLinePlot(
                        y_data=[[], [], []],
                        y_min=-3.0,
                        y_max=3.0,
                        plot_height=150,
                        show_legend=True,
                        legends=["Vx (m/s)", "Vy (m/s)", "Vz (m/s)"],
                        max_datapoints=max_points,
                    )
                
                # Angular velocity plot
                with omni.ui.CollapsableFrame("Angular Velocity (rad/s)", collapsed=False):
                    self.ang_vel_plot = LiveLinePlot(
                        y_data=[[], [], []],
                        y_min=-5.0,
                        y_max=5.0,
                        plot_height=150,
                        show_legend=True,
                        legends=["Wx (rad/s)", "Wy (rad/s)", "Wz (rad/s)"],
                        max_datapoints=max_points,
                    )
    
    def update(self, pos, rot, lin_vel, ang_vel, dt, motion_info=None):
        """Update the plots with new data.
        
        Args:
            pos: Position tensor (3,)
            rot: Rotation quaternion tensor (4,) in (w,x,y,z) format
            lin_vel: Linear velocity tensor (3,)
            ang_vel: Angular velocity tensor (3,)
            dt: Time step
            motion_info: Optional dict with 'motion_name', 'motion_idx', 'total_motions', 'frame_idx', 'total_frames'
        """
        # Update status labels if motion info is provided
        if motion_info is not None and hasattr(self, 'motion_label'):
            self.motion_label.text = f"Playing motion {motion_info['motion_idx']}/{motion_info['total_motions']}: {motion_info['motion_name']}"
            self.frame_label.text = f"Frame: {motion_info['frame_idx']} / {motion_info['total_frames']}"
        
        # Convert tensors to numpy for plotting
        pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
        rot_np = rot.cpu().numpy() if isinstance(rot, torch.Tensor) else rot
        lin_vel_np = lin_vel.cpu().numpy() if isinstance(lin_vel, torch.Tensor) else lin_vel
        ang_vel_np = ang_vel.cpu().numpy() if isinstance(ang_vel, torch.Tensor) else ang_vel
        
        # Add data points to each plot
        self.pos_plot.add_datapoint([pos_np[0], pos_np[1], pos_np[2]])
        self.rot_plot.add_datapoint([rot_np[0], rot_np[1], rot_np[2], rot_np[3]])
        self.lin_vel_plot.add_datapoint([lin_vel_np[0], lin_vel_np[1], lin_vel_np[2]])
        self.ang_vel_plot.add_datapoint([ang_vel_np[0], ang_vel_np[1], ang_vel_np[2]])
    
    def clear(self):
        """Clear all plots."""
        self.pos_plot.clear()
        self.rot_plot.clear()
        self.lin_vel_plot.clear()
        self.ang_vel_plot.clear()
    
    def destroy(self):
        """Clean up the plotter."""
        if hasattr(self, 'window') and self.window is not None:
            self.window.visible = False


def reorder_from_isaacgym_to_isaacsim(joint_tensor):
    """Convert joint ordering from Isaac Gym (depth-first) to Isaac Sim (breadth-first).
    
    Isaac Gym order (AMP data): [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, 
                                   RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
    Isaac Sim order: [FR_hip, FL_hip, RR_hip, RL_hip, FR_thigh, FL_thigh, 
                      RR_thigh, RL_thigh, FR_calf, FL_calf, RR_calf, RL_calf]
    
    Args:
        joint_tensor: Tensor of shape (..., 12) in Isaac Gym ordering
        
    Returns:
        Tensor of shape (..., 12) in Isaac Sim ordering
    """
    # Convert to a 4x3 tensor (4 legs x 3 joints per leg)
    reshaped_tensor = torch.reshape(joint_tensor, (-1, 4, 3))
    # Transpose to get 3x4 (3 joint types x 4 legs)
    transposed_tensor = torch.transpose(reshaped_tensor, 1, 2)
    # Flatten back to 1D
    rearranged_tensor = torch.reshape(transposed_tensor, (-1, 12))
    return rearranged_tensor


def analyze_angular_velocity_integration(motion_files, all_frames, frame_dt):
    """Analyze and plot integrated angular velocities for each motion.
    
    Args:
        motion_files: List of motion file paths
        all_frames: List of frame tensors for each motion
        frame_dt: Time step between frames
    """
    num_motions = len(all_frames)
    
    # Create output directory
    output_dir = 'outputs/angular_velocity_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create combined figure with all motions
    fig_combined = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, figure=fig_combined, hspace=0.3)
    
    axes_combined = [
        fig_combined.add_subplot(gs[0, 0]),  # Roll (Wx)
        fig_combined.add_subplot(gs[1, 0]),  # Pitch (Wy)
        fig_combined.add_subplot(gs[2, 0]),  # Yaw (Wz)
    ]
    
    axis_names = ['Roll (Wx)', 'Pitch (Wy)', 'Yaw (Wz)']
    colors = plt.cm.tab20(np.linspace(0, 1, num_motions))
    
    print("\n" + "="*80)
    print("Angular Velocity Integration Analysis")
    print("="*80)
    
    individual_figs = []
    
    # Process each motion
    for motion_idx, (frames, motion_file) in enumerate(zip(all_frames, motion_files)):
        motion_name = os.path.basename(motion_file).replace('.txt', '')
        num_frames = frames.shape[0]
        
        # Extract angular velocities
        ang_vels = frames[:, AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]  # (num_frames, 3)
        ang_vels_np = ang_vels.cpu().numpy()
        
        # Integrate angular velocities (cumulative sum * dt)
        integrated_angles = np.cumsum(ang_vels_np, axis=0) * frame_dt
        
        # Convert to degrees for better readability
        integrated_angles_deg = np.degrees(integrated_angles)
        
        # Time array
        time = np.arange(num_frames) * frame_dt
        
        # Print statistics
        print(f"\nMotion {motion_idx+1}: {motion_name}")
        print(f"  Duration: {time[-1]:.2f}s, Frames: {num_frames}")
        print(f"  Final integrated angles (degrees):")
        print(f"    Roll  (Wx): {integrated_angles_deg[-1, 0]:7.2f}°")
        print(f"    Pitch (Wy): {integrated_angles_deg[-1, 1]:7.2f}°")
        print(f"    Yaw   (Wz): {integrated_angles_deg[-1, 2]:7.2f}°")
        print(f"  Max angular velocities (rad/s):")
        print(f"    Wx: {np.max(np.abs(ang_vels_np[:, 0])):.3f}")
        print(f"    Wy: {np.max(np.abs(ang_vels_np[:, 1])):.3f}")
        print(f"    Wz: {np.max(np.abs(ang_vels_np[:, 2])):.3f}")
        
        # Plot on combined figure
        for axis_idx in range(3):
            axes_combined[axis_idx].plot(
                time, 
                integrated_angles_deg[:, axis_idx],
                label=motion_name,
                color=colors[motion_idx],
                linewidth=1.5,
                alpha=0.8
            )
        
        # Create individual figure for this motion
        fig_individual = plt.figure(figsize=(12, 8))
        gs_ind = GridSpec(3, 1, figure=fig_individual, hspace=0.3)
        
        axes_individual = [
            fig_individual.add_subplot(gs_ind[0, 0]),  # Roll (Wx)
            fig_individual.add_subplot(gs_ind[1, 0]),  # Pitch (Wy)
            fig_individual.add_subplot(gs_ind[2, 0]),  # Yaw (Wz)
        ]
        
        # Plot individual motion data
        for axis_idx in range(3):
            ax = axes_individual[axis_idx]
            
            # Plot integrated angle
            ax.plot(
                time, 
                integrated_angles_deg[:, axis_idx],
                color=colors[motion_idx],
                linewidth=2,
                label='Integrated Angle'
            )
            
            # Configure individual subplot
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Integrated Angle (°)', fontsize=11)
            ax.set_title(f'{axis_names[axis_idx]} Integration', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            
            # Add statistics text
            final_angle = integrated_angles_deg[-1, axis_idx]
            max_vel = np.max(np.abs(ang_vels_np[:, axis_idx]))
            stats_text = f'Final: {final_angle:.2f}°\nMax ω: {max_vel:.3f} rad/s'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Configure individual figure
        fig_individual.suptitle(f'Angular Velocity Integration: {motion_name}', 
                               fontsize=14, fontweight='bold', y=0.995)
        
        # Save individual figure
        individual_path = os.path.join(output_dir, f'{motion_name}_integration.png')
        fig_individual.savefig(individual_path, dpi=150, bbox_inches='tight')
        print(f"  Saved individual plot: {individual_path}")
        
        individual_figs.append(fig_individual)
    
    # Configure combined plots
    for axis_idx, (ax, name) in enumerate(zip(axes_combined, axis_names)):
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(f'Integrated Angle (°)', fontsize=10)
        ax.set_title(f'{name} Integration', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    fig_combined.suptitle('Angular Velocity Integration Analysis for All Motions', 
                         fontsize=14, fontweight='bold', y=0.995)
    
    # Save combined figure
    combined_path = os.path.join(output_dir, 'all_motions_combined.png')
    fig_combined.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Combined plot saved to: {combined_path}")
    print(f"Individual plots saved to: {output_dir}/")
    print(f"{'='*80}\n")
    
    # Close all figures (don't display)
    plt.close(fig_combined)
    for fig in individual_figs:
        plt.close(fig)
    
    return None, None


def main():
    """Main execution function."""
    
    # Get all motion files from directory
    if not os.path.exists(MOTION_DIR):
        print(f"\nError: Motion directory not found: {MOTION_DIR}")
        simulation_app.close()
        return
    
    motion_files = sorted(glob.glob(os.path.join(MOTION_DIR, "*.txt")))
    
    if not motion_files:
        print(f"\nError: No motion files found in {MOTION_DIR}")
        simulation_app.close()
        return
    
    print(f"\nFound {len(motion_files)} motion files:")
    for f in motion_files:
        print(f"  - {os.path.basename(f)}")
    
    print(f"\nLoading all motion files from: {MOTION_DIR}")
    
    # Load motion data using AMPLoader
    # AMPLoader expects device and time_between_frames
    device = args_cli.device if args_cli.device else "cuda:0"
    
    # Load all motions
    motion_loader = AMPLoader(
        device=device,
        time_between_frames=0.02,  # Will be updated from actual motion data
        data_dir="",
        preload_transitions=False,
        motion_files=motion_files
    )
    
    # Get motion data from all trajectories
    all_frames = motion_loader.trajectories_full  # List of tensors
    num_motions = len(all_frames)
    frame_dt = motion_loader.trajectory_frame_durations[0]  # Assume all have same dt
    
    print(f"\nLoaded {num_motions} motions with dt={frame_dt:.4f}s")
    for i, frames in enumerate(all_frames):
        duration = frames.shape[0] * frame_dt
        print(f"  Motion {i+1}: {frames.shape[0]} frames, {duration:.2f}s - {os.path.basename(motion_files[i])}")
    
    # Analyze and plot angular velocity integration
    _, _ = analyze_angular_velocity_integration(motion_files, all_frames, frame_dt)
    
    # Configure simulation with dt matching motion dt
    sim_cfg = sim_utils.SimulationCfg(
        dt=frame_dt,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        render_interval=1,
        enable_scene_query_support=True,
        use_fabric=True,
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # TGS solver
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=4,
            max_velocity_iteration_count=4,
            enable_ccd=True,
            enable_stabilization=True,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025
        ),
    )
    
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.5])
    
    # Configure scene
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene = InteractiveScene(scene_cfg)
    
    # Add lighting
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    
    # Add ground plane
    ground_cfg = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1))
    ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, 0.0))
    
    # Reset simulation
    sim.reset()
    scene.reset()
    
    robot = scene["robot"]
    env_id = torch.tensor([0], device=device)
    
    # Create real-time plotter
    plotter = RealTimePlotter(max_points=500)
    
    print("\nStarting Go2 AMP motion replay...")
    print("Real-time plotting windows opened")
    print("Tip: Close window or press Ctrl+C to exit\n")
    
    try:
        frame_count = 0
        motion_idx = 0  # Start with first motion
        
        # Fixed rotation: identity quaternion (w=1, x=0, y=0, z=0) for when FIX_BASE_ROTATION=True
        fixed_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device) if FIX_BASE_ROTATION else None
        
        while simulation_app.is_running():
            # Get current motion frames
            frames = all_frames[motion_idx]
            num_frames = frames.shape[0]
            motion_name = os.path.basename(motion_files[motion_idx])
            
            print(f"\nPlaying motion {motion_idx+1}/{num_motions}: {motion_name}")
            
            for i in range(num_frames):
                if not simulation_app.is_running():
                    break
                
                # Get current frame
                frame = frames[i]
                
                # Extract state from AMP frame
                # AMP format: [root_pos(3), root_rot(4), joint_pos(12), tar_toe_pos(12), 
                #              lin_vel(3), ang_vel(3), joint_vel(12), tar_toe_vel(12)]
                root_pos = frame[AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX].unsqueeze(0)
                root_rot_amp = frame[AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX].unsqueeze(0)
                joint_pos = frame[AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX].unsqueeze(0)
                lin_vel = frame[AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX].unsqueeze(0)
                ang_vel = frame[AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX].unsqueeze(0)
                joint_vel = frame[AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX].unsqueeze(0)
                
                # Reorder joints from Isaac Gym (AMP data) to Isaac Sim format
                # joint_pos = reorder_from_isaacgym_to_isaacsim(joint_pos)
                # joint_vel = reorder_from_isaacgym_to_isaacsim(joint_vel)
                
                # Construct root state: [pos(3), rot(4), lin_vel(3), ang_vel(3)]
                # Use fixed rotation or AMP rotation based on FIX_BASE_ROTATION setting
                root_rot = fixed_rot if FIX_BASE_ROTATION else root_rot_amp
                root_state = torch.cat([root_pos, root_rot, lin_vel, ang_vel], dim=-1)
                
                # Write state to simulation
                robot.write_root_link_pose_to_sim(root_state[:, :7], env_id)
                robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_id)
                robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_id)
                
                # Update plotter with current state
                plotter.update(
                    pos=root_pos[0],      # (3,)
                    rot=root_rot[0],      # (4,) quaternion (fixed or from AMP data)
                    lin_vel=lin_vel[0],   # (3,)
                    ang_vel=ang_vel[0],   # (3,)
                    dt=frame_dt,
                    motion_info={
                        'motion_name': motion_name,
                        'motion_idx': motion_idx + 1,
                        'total_motions': num_motions,
                        'frame_idx': i + 1,
                        'total_frames': num_frames,
                    }
                )
                
                # Step simulation
                scene.update(dt=sim.get_physics_dt())
                scene.write_data_to_sim()
                sim.step(render=True)
                
                frame_count += 1
                
                # Print progress occasionally
                if frame_count % 100 == 0:
                    print(f"Total frames played: {frame_count}", end='\r', flush=True)
            
            # Move to next motion
            motion_idx = (motion_idx + 1) % num_motions
            
            # Exit if not looping
            if not args_cli.loop and motion_idx == 0:
                print(f"\nCompleted playback of all {num_motions} motions ({frame_count} total frames)")
                break
                
    except KeyboardInterrupt:
        print("\n\nPlayback interrupted by user")
    
    finally:
        print("\nClosing simulation...")
        plotter.destroy()
        simulation_app.close()


if __name__ == "__main__":
    main()
