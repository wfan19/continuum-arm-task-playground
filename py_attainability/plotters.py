import plotly.graph_objects as go
import numpy as np

from py_attainability.lie import Pose2

def plot_poses(mat_poses, fig=go.Figure()):
    body_points_axes = np.zeros([2, 2, 3*mat_poses.shape[0]])
    axis_length=0.1
    idx = 0
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
        
    axis_colors = ["#D81B60", "#00BF9F", "#278DE6"]
    axis_names = ["X", "Y"]
    marker_size=3
    for i_axis, axis in enumerate(body_points_axes):
        fig.add_trace(go.Scatter(
            x = axis[0, :],
            y = axis[1, :],
            name=axis_names[i_axis],
            mode="lines+markers",
            showlegend=False,
            marker=dict(color=axis_colors[i_axis], size=marker_size)
        ))

    xs = mat_poses[:, 0, 2]
    ys = mat_poses[:, 1, 2]
    fig.add_trace(go.Scatter(
        x=xs,
        y = ys,
        mode="markers",
        name="Poses"
    ))

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
