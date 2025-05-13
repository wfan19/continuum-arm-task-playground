import plotly.graph_objects as go
import numpy as np

from py_attainability.lie import Pose2

def plot_poses(mat_poses, fig=go.Figure()):
    body_points_axes = np.zeros([2, 2, 3*mat_poses.shape[0]])
    axis_length=0.05
    idx = 0
    # Prepare data for plotting the axes
    for i, pose in enumerate(mat_poses):
        # Iterate through poses. For each pose:
        #   - Add its position to each body_points_axes
        #   - Add its the basis vector for each dimension/direction
        #   - Add a None vector as a spacer
        for i_axis, body_points_axis in enumerate(body_points_axes):
            # Append the position of the pose
            body_points_axis[:, idx] = Pose2.translation(pose)

            # Append x or y unit vector 
            body_points_axis[:, idx+1] = Pose2.translation(pose) + pose[0:2, i_axis] * axis_length

            # Append a None vector as a gap
            body_points_axis[:, idx+2] = np.array([None, None])

        idx += 3
        
    axis_colors = ["#ee2b8b", "#2db67d", "#648fff"]
    axis_names = ["X", "Y"]
    quiver_size=7
    marker_size=12
    linewidth=4

    xs = mat_poses[:, 0, 2]
    ys = mat_poses[:, 1, 2]

    fig.add_trace(go.Scatter(
        x=xs,
        y = ys,
        mode="markers+lines",
        name="Poses",
        legendgroup="backbone",
        line=dict(color="darkgray", width=linewidth),
        marker=dict(size=marker_size, color="gray")
    ))

    axis_name = ["X", "Y", "Z"]
    for i_axis, axis in enumerate(body_points_axes):
        fig.add_trace(go.Scatter(
            x = axis[0, :],
            y = axis[1, :],
            name=f"Transforms",
            mode="lines+markers",
            showlegend=True if i_axis == 0 else False,
            legendgroup="transforms",
            marker=dict(color=axis_colors[i_axis], size=quiver_size)
        ))

    fig.add_trace(go.Scatter(
        x=xs,
        y = ys,
        mode="markers",
        name="Poses",
        legendgroup="backbone",
        showlegend=False,
        marker=dict(size=marker_size, color="darkgray")
    ))

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=800)
    return fig
