import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import textwrap
import os

#################################################################################
#                                   Data Params                                 #
#################################################################################
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]

kit_raw_offsets = np.array(
    [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
     [1, 0, 0], [0, -1, 0], [0, -1, 0], [-1, 0, 0], [0, -1, 0],
     [0, -1, 0], [1, 0, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1],
     [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1],
     [0, 0, 1]])
t2m_raw_offsets = np.array([[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0],
                            [0,-1,0], [0,1,0], [0,-1,0], [0,-1,0], [0,1,0],
                            [0,0,1], [0,0,1], [0,1,0], [1,0,0], [-1,0,0],
                            [0,0,1], [0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0],
                            [0,-1,0], [0,-1,0]])

#################################################################################
#                                  Joints Revert                                #
#################################################################################
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

#################################################################################
#                                 Motion Plotting                               #
#################################################################################
# def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
#     matplotlib.use('Agg')

#     title_sp = title.split(' ')
#     if len(title_sp) > 20:
#         title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
#     elif len(title_sp) > 10:
#         title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

#     def init():
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([0, radius])
#         ax.set_zlim3d([0, radius])
#         fig.suptitle(title, fontsize=20)
#         ax.grid(b=False)

#     def plot_xzPlane(minx, maxx, miny, minz, maxz):
#         verts = [
#             [minx, miny, minz],
#             [minx, miny, maxz],
#             [maxx, miny, maxz],
#             [maxx, miny, minz]
#         ]
#         xz_plane = Poly3DCollection([verts])
#         xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)

#     data = joints.copy().reshape(len(joints), -1, 3)
#     fig = plt.figure(figsize=figsize)
#     ax = p3.Axes3D(fig)
#     init()
#     MINS = data.min(axis=0).min(axis=0)
#     MAXS = data.max(axis=0).max(axis=0)
#     colors = ['red', 'blue', 'black', 'red', 'blue',
#               'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
#               'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
#     frame_number = data.shape[0]

#     height_offset = MINS[1]
#     data[:, :, 1] -= height_offset
#     trajec = data[:, 0, [0, 2]]

#     data[..., 0] -= data[:, 0:1, 0]
#     data[..., 2] -= data[:, 0:1, 2]

#     # def update(index):
#     #     ax.lines = []
#     #     ax.collections = []
#     #     ax.view_init(elev=120, azim=-90)
#     #     ax.dist = 7.5
#     #     plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
#     #                  MAXS[2] - trajec[index, 1])

#     #     if index > 1:
#     #         ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
#     #                   trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
#     #                   color='blue')

#     #     for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
#     #         if i < 5:
#     #             linewidth = 4.0
#     #         else:
#     #             linewidth = 2.0
#     #         ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
#     #                   color=color)

#     #     plt.axis('off')
#     #     ax.set_xticklabels([])
#     #     ax.set_yticklabels([])
#     #     ax.set_zticklabels([])

#     def update(index):
#         ax.clear()  # This replaces ax.lines = [] and ax.collections = []
        
#         # Re-apply axis properties after clearing
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([0, radius])
#         ax.set_zlim3d([0, radius])
#         ax.grid(b=False)

#         # The rest of the original update function
#         ax.view_init(elev=120, azim=-90)
#         ax.dist = 7.5
#         plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
#                      MAXS[2] - trajec[index, 1])

#         if index > 1:
#             ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
#                       trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
#                       color='blue')

#         for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
#             if i < 5:
#                 linewidth = 4.0
#             else:
#                 linewidth = 2.0
#             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
#                       color=color)

#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])

#     ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

#     ani.save(save_path, fps=fps)
#     plt.close()

def plot_3d_motion_gif(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=20, radius=4, text_prompt=None, style_label=None):
    """
    Renders a 3D motion sequence and saves it as a GIF.
    """
    matplotlib.use('Agg')
    
    # =========================================================================
    # Process title (wrap long titles)
    # =========================================================================
    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    data = joints.copy().reshape(len(joints), -1, 3)
    
    # =========================================================================
    # Calculate figure size based on whether we have overlay text
    # =========================================================================
    has_overlay = text_prompt or style_label
    
    if has_overlay:
        # Add extra height at top for overlay text
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 1.0))
        plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95)
    
    ax = fig.add_subplot(111, projection='3d')

    # =========================================================================
    # Add overlay text at TOP of figure
    # =========================================================================
    if has_overlay:
        overlay_lines = []
        if text_prompt:
            # Wrap long prompts
            if len(text_prompt) > 60:
                wrapped = textwrap.fill(text_prompt, width=60)
                overlay_lines.append(f"Prompt: {wrapped}")
            else:
                overlay_lines.append(f"Prompt: {text_prompt}")
        if style_label:
            overlay_lines.append(f"Style: {style_label}")
        
        # Join with newline for cleaner display
        overlay_text = "\n".join(overlay_lines)
        
        # Place at top of figure
        fig.text(0.5, 0.97, overlay_text, ha='center', va='top', 
                 fontsize=11, fontweight='normal', family='sans-serif',
                 linespacing=1.5)

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # Title below the overlay text
        ax.set_title(title, fontsize=14, pad=10)
        ax.grid(b=False)

    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    # Normalize data for consistent visualization
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    # Center the animation at the origin for every frame
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    def update(index):
        ax.clear()  # Clear previous frame
       
        # Re-apply axis properties after clearing
        init()

        # Set consistent viewing angle
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
       
        # Plot ground plane and trajectory
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')

        # Plot the skeleton
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer=writer)
    plt.close()
   
    return save_path


def plot_3d_motion_side_by_side(save_path, kinematic_tree, joints1, joints2, title1, title2, figsize=(12, 6), fps=20, radius=4, text_prompt=None, style_label=None):    
    """
    Renders two 3D motion sequences side-by-side in a single GIF.
    """
    matplotlib.use('Agg')
    
    # =========================================================================
    # Calculate figure height based on whether we have overlay text
    # =========================================================================
    has_overlay = text_prompt or style_label
    
    if has_overlay:
        # Add extra height at top for text
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 0.8))
        # Reserve space at top for text (leaves 0.92 of figure for plots)
        plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    axes = [ax1, ax2]
    all_joints = [joints1, joints2]
    titles = [title1, title2]
    
    # =========================================================================
    # Add overlay text at TOP of figure (not bottom)
    # =========================================================================
    if has_overlay:
        overlay_lines = []
        if text_prompt:
            # Wrap long prompts
            if len(text_prompt) > 80:
                wrapped = textwrap.fill(text_prompt, width=80)
                overlay_lines.append(f"Prompt: {wrapped}")
            else:
                overlay_lines.append(f"Prompt: {text_prompt}")
        if style_label:
            overlay_lines.append(f"Style: {style_label}")
        
        overlay_text = " | ".join(overlay_lines) if len(overlay_lines) > 1 and not '\n' in overlay_lines[0] else "\n".join(overlay_lines)
        
        fig.suptitle(overlay_text, fontsize=11, fontweight='normal', 
                     y=0.96, ha='center', va='top',
                     family='sans-serif')
    
    # ... rest of your code remains the same ...
    all_data = []
    all_trajec = []
    all_MINS = []
    all_MAXS = []
    for joints in all_joints:
        data = joints.copy().reshape(len(joints), -1, 3)
        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)
        
        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]
        
        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]
        
        all_data.append(data)
        all_trajec.append(trajec)
        all_MINS.append(MINS)
        all_MAXS.append(MAXS)
        
    def init_ax(ax, title):
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(b=False)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
    def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
        
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    
    def update(index):
        for i, ax in enumerate(axes):
            ax.clear()
            init_ax(ax, titles[i])
            ax.view_init(elev=120, azim=-90)
            ax.dist = 7.5
            trajec = all_trajec[i]
            MINS = all_MINS[i]
            MAXS = all_MAXS[i]
            data = all_data[i]
            plot_xzPlane(ax, MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                         MAXS[2] - trajec[index, 1])
            if index > 1:
                ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                          trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                          color='blue')
            for j, (chain, color) in enumerate(zip(kinematic_tree, colors)):
                linewidth = 4.0 if j < 5 else 2.0
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                          color=color)
                          
    frame_number = len(joints1)
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    writer = PillowWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()


def plot_3d_motion_three_way(save_path, kinematic_tree, joints1, joints2, joints3, title1, title2, title3, figsize=(18, 6), fps=20, radius=4):
    """
    Renders three 3D motion sequences side-by-side in a single GIF.
    """
    matplotlib.use('Agg')

    # Create a figure with three subplots
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    axes = [ax1, ax2, ax3]
    all_joints = [joints1, joints2, joints3]
    titles = [title1, title2, title3]
    
    all_data = []
    all_trajec = []
    all_MINS = []
    all_MAXS = []

    for joints in all_joints:
        data = joints.copy().reshape(len(joints), -1, 3)
        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)
        
        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]
        
        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]
        
        all_data.append(data)
        all_trajec.append(trajec)
        all_MINS.append(MINS)
        all_MAXS.append(MAXS)

    def init_ax(ax, title):
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        ax.set_title(title, fontsize=12)
        ax.grid(b=False)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    
    def update(index):
        for i, ax in enumerate(axes):
            ax.clear()
            init_ax(ax, titles[i])
            ax.view_init(elev=120, azim=-90)
            ax.dist = 7.5

            trajec = all_trajec[i]
            MINS = all_MINS[i]
            MAXS = all_MAXS[i]
            data = all_data[i]

            plot_xzPlane(ax, MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                         MAXS[2] - trajec[index, 1])

            if index > 1:
                ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                          trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                          color='blue')

            for j, (chain, color) in enumerate(zip(kinematic_tree, colors)):
                linewidth = 4.0 if j < 5 else 2.0
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                          color=color)

    frame_number = len(joints1)
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    writer = PillowWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()


def plot_3d_motion_three_way2(save_path, kinematic_tree, joints1, joints2, joints3, title1, title2, title3, figsize=(18, 6), fps=20, radius=4, text_prompt=None, style_label=None):
    """
    Renders three 3D motion sequences side-by-side in a single GIF.
    """
    matplotlib.use('Agg')

    # =========================================================================
    # Calculate figure height based on whether we have overlay text
    # =========================================================================
    has_overlay = text_prompt or style_label
    
    if has_overlay:
        # Add extra height at top for text
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 0.8))
        # Reserve space at top for text
        plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    axes = [ax1, ax2, ax3]
    all_joints = [joints1, joints2, joints3]
    titles = [title1, title2, title3]
    
    # =========================================================================
    # Add overlay text at TOP of figure
    # =========================================================================
    if has_overlay:
        overlay_lines = []
        if text_prompt:
            # Wrap long prompts - increased width to 120 since 3-way fig is wider
            if len(text_prompt) > 120:
                wrapped = textwrap.fill(text_prompt, width=120)
                overlay_lines.append(f"Prompt: {wrapped}")
            else:
                overlay_lines.append(f"Prompt: {text_prompt}")
        if style_label:
            overlay_lines.append(f"Style: {style_label}")
        
        overlay_text = " | ".join(overlay_lines) if len(overlay_lines) > 1 and not '\n' in overlay_lines[0] else "\n".join(overlay_lines)
        
        fig.suptitle(overlay_text, fontsize=12, fontweight='normal', 
                     y=0.96, ha='center', va='top',
                     family='sans-serif')

    all_data = []
    all_trajec = []
    all_MINS = []
    all_MAXS = []

    for joints in all_joints:
        data = joints.copy().reshape(len(joints), -1, 3)
        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)
        
        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]
        
        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]
        
        all_data.append(data)
        all_trajec.append(trajec)
        all_MINS.append(MINS)
        all_MAXS.append(MAXS)

    def init_ax(ax, title):
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        ax.set_title(title, fontsize=12, pad=10) # Added padding so sub-titles don't clip
        ax.grid(False) # Fixed deprecation warning (changed from b=False)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    
    def update(index):
        for i, ax in enumerate(axes):
            ax.clear()
            init_ax(ax, titles[i])
            ax.view_init(elev=120, azim=-90)
            ax.dist = 7.5

            trajec = all_trajec[i]
            MINS = all_MINS[i]
            MAXS = all_MAXS[i]
            data = all_data[i]

            plot_xzPlane(ax, MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                         MAXS[2] - trajec[index, 1])

            if index > 1:
                ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                          trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                          color='blue')

            for j, (chain, color) in enumerate(zip(kinematic_tree, colors)):
                linewidth = 4.0 if j < 5 else 2.0
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                          color=color)

    frame_number = len(joints1)
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    writer = PillowWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()