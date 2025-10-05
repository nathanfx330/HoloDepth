import torch
import torch.nn.functional as F

def create_psv(src_imgs, src_poses, src_intrinsics, ref_pose, ref_intrinsics, plane_depths):
    """
    Creates a Plane Sweep Volume (PSV) by warping source views to a reference view.
    """
    B, V, C, H, W = src_imgs.shape
    D = len(plane_depths)
    device = src_imgs.device

    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
    pixel_coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=-1).reshape(1, H * W, 3, 1).expand(B, -1, -1, -1)

    K_ref_inv = torch.inverse(ref_intrinsics)
    cam_rays = K_ref_inv.reshape(B, 1, 3, 3) @ pixel_coords
    
    psv = torch.zeros(B, V, D, H, W, C, device=device)

    for d_idx, depth in enumerate(plane_depths):
        points_in_ref_cam = cam_rays * depth
        points_homo = torch.cat([points_in_ref_cam, torch.ones(B, H * W, 1, 1, device=device)], dim=2)
        
        for v_idx in range(V):
            K_src = src_intrinsics[:, v_idx, :, :]
            T_src = src_poses[:, v_idx, :, :]
            T_src_inv = torch.inverse(T_src)
            T_ref_to_src = T_src_inv @ ref_pose[:, 0, :, :]
            
            projected_points_homo = T_ref_to_src.reshape(B, 1, 4, 4) @ points_homo
            projected_points_cam = projected_points_homo[:, :, :3, :]
            projected_pixel_coords = K_src.reshape(B, 1, 3, 3) @ projected_points_cam
            
            u = projected_pixel_coords[:, :, 0, :] / (projected_pixel_coords[:, :, 2, :] + 1e-8)
            v = projected_pixel_coords[:, :, 1, :] / (projected_pixel_coords[:, :, 2, :] + 1e-8)

            u_norm = (u / (W - 1)) * 2 - 1
            v_norm = (v / (H - 1)) * 2 - 1
            
            grid = torch.stack([u_norm, v_norm], dim=-1).reshape(B, H, W, 2)

            warped_img = F.grid_sample(src_imgs[:, v_idx, :, :, :], grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            
            psv[:, v_idx, d_idx, :, :, :] = warped_img.permute(0, 2, 3, 1)

    return psv


# ======================================================================
# --- THE REAL FIX: A completely rewritten, functional MPI renderer ---
def render_mpi(mpi_rgba, ref_pose, ref_intrinsics, tgt_pose, tgt_intrinsics, plane_depths):
    """
    Renders a novel view from an MPI by warping each plane to the target
    view and then alpha compositing them.
    """
    B, D, H, W, _ = mpi_rgba.shape
    device = mpi_rgba.device

    # Create a grid of pixel coordinates for the target image
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
    pixel_coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=-1).view(1, H * W, 3, 1).expand(B, -1, -1, -1)

    # Inverse of target camera intrinsics to cast rays from pixels
    K_tgt_inv = torch.inverse(tgt_intrinsics)
    cam_rays = K_tgt_inv @ pixel_coords

    final_image = torch.zeros(B, H, W, 3, device=device)
    transmittance = torch.ones(B, H, W, 1, device=device)

    # Transformation from target camera space to reference camera space
    T_tgt_to_ref = torch.inverse(ref_pose) @ tgt_pose

    for d_idx, depth in enumerate(plane_depths):
        # Calculate where the target rays intersect the current MPI plane in the target camera's space
        points_in_tgt_cam = cam_rays * depth
        points_homo_tgt = torch.cat([points_in_tgt_cam, torch.ones(B, H * W, 1, 1, device=device)], dim=2)

        # Transform these intersection points into the reference camera's coordinate system
        projected_points_homo_ref = T_tgt_to_ref @ points_homo_tgt
        projected_points_ref_cam = projected_points_homo_ref[:, :, :3, :]

        # Project the 3D points onto the 2D MPI plane using the reference camera's intrinsics
        K_ref = ref_intrinsics
        projected_pixel_coords = K_ref @ projected_points_ref_cam

        # Normalize the resulting 2D coordinates to create the sampling grid for grid_sample
        u = projected_pixel_coords[:, :, 0, :] / (projected_pixel_coords[:, :, 2, :] + 1e-8)
        v = projected_pixel_coords[:, :, 1, :] / (projected_pixel_coords[:, :, 2, :] + 1e-8)
        
        u_norm = (u / (W - 1)) * 2 - 1
        v_norm = (v / (H - 1)) * 2 - 1
        
        grid = torch.stack([u_norm, v_norm], dim=-1).view(B, H, W, 2)

        # Sample (warp) the current MPI plane using the grid
        current_plane_rgba = mpi_rgba[:, d_idx].permute(0, 3, 1, 2)
        warped_plane = F.grid_sample(current_plane_rgba, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_plane = warped_plane.permute(0, 2, 3, 1)

        # Alpha composite the correctly warped plane
        layer_rgb = warped_plane[..., :3]
        layer_alpha = warped_plane[..., 3:]
        
        contribution = transmittance * layer_alpha * layer_rgb
        final_image += contribution
        transmittance = transmittance * (1.0 - layer_alpha)

    # Add a white background where transmittance is still high
    final_image += transmittance * torch.ones_like(final_image)

    return final_image
# ======================================================================