import torch
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import os
import tqdm
import argparse
import random
import json
from PIL import Image

# --- Custom Modules ---
import data_loader, utils, model, vgg

class SpacesIterableDataset(IterableDataset):
    def __init__(self, scenes_base_path, scene_list, config):
        super(SpacesIterableDataset).__init__()
        self.scenes_base_path, self.scene_list, self.config = scenes_base_path, scene_list, config
        self.scene_data_cache = {}
    def __iter__(self):
        while True:
            scene_name = random.choice(self.scene_list)
            try:
                if scene_name not in self.scene_data_cache:
                    scene_data = data_loader.load_scene_data(self.scenes_base_path, scene_name, self.config['resolution_w'], self.config['resolution_h'])
                    if not scene_data: continue
                    self.scene_data_cache[scene_name] = scene_data
                else: scene_data = self.scene_data_cache[scene_name]
                
                # --- MODIFIED: Use the specified ref_view and witness_views ---
                s_views = [scene_data[i] for i in self.config['source_view_indices']]
                v_views = [scene_data[i] for i in self.config['witness_view_indices']]
                r_view = scene_data[self.config['ref_view_idx']] # Use the specific reference view

                s_imgs, s_poses, s_intr = torch.stack([v['image'] for v in s_views]), torch.stack([v['pose'] for v in s_views]), torch.stack([v['intrinsics'] for v in s_views])
                v_imgs, v_poses, v_intr = torch.stack([v['image'] for v in v_views]), torch.stack([v['pose'] for v in v_views]), torch.stack([v['intrinsics'] for v in v_views])
                yield s_imgs, s_poses, s_intr, v_imgs, v_poses, v_intr, r_view['pose'], r_view['intrinsics']
            except Exception as e:
                print(f"\n[Warning] Skipping during iteration: {e}"); continue

# [save_snapshot_image function remains unchanged]
def save_snapshot_image(mpi_rgba, ref_pose, ref_intr, val_imgs, val_poses, val_intr, planes, global_step, config, output_base_dir):
    output_dir = os.path.join(output_base_dir, "training_progress"); os.makedirs(output_dir, exist_ok=True)
    num_val_views = len(config['witness_view_indices']); comp_img = Image.new('RGB', (config['resolution_w'] * num_val_views, config['resolution_h'] * 2))
    try:
        for i in range(num_val_views):
            v_pose, v_intr = val_poses[:, i].unsqueeze(1), val_intr[:, i].unsqueeze(1)
            rendered_view = utils.render_mpi(mpi_rgba, ref_pose.unsqueeze(1), ref_intr.unsqueeze(1), v_pose, v_intr, planes)
            rendered_np = ((rendered_view[0].detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
            rendered_pil = Image.fromarray(rendered_np); gt_np = ((val_imgs[0, i].detach().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            gt_pil = Image.fromarray(gt_np); comp_img.paste(gt_pil, (config['resolution_w'] * i, 0)); comp_img.paste(rendered_pil, (config['resolution_w'] * i, config['resolution_h']))
        img_path = os.path.join(output_dir, f"step_{global_step:06d}.png"); comp_img.save(img_path)
    except Exception as e: print(f"\n[Warning: Could not save snapshot image for step {global_step}. Error: {e}]")


def main():
    parser = argparse.ArgumentParser(description="Train the HoloDepth Model with artist-controlled parameters.")
    parser.add_argument('--scenes_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='training_output')
    parser.add_argument('--num_planes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    # --- NEW: Artist-controlled view selection ---
    parser.add_argument('--ref_view', type=int, default=0, help="Index of the camera to use as the MPI's reference 'straight-on' view.")
    parser.add_argument('--witness_views', nargs=3, type=int, default=[10, 72, 135], help="Three camera indices to use for validation snapshots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"--- Output will be saved to: '{args.output_dir}' ---")

    config = {
        'resolution_w': 200, 'resolution_h': 120, 'epochs': args.epochs, 'batch_size': 4,
        'source_view_indices': [0, 119], # Keep wide baseline for strong depth signal
        'witness_view_indices': args.witness_views,
        'ref_view_idx': args.ref_view,
        'checkpoint_path': os.path.join(args.output_dir, 'hybrid_mpi_checkpoint.pt'),
        'learning_rate': 1e-5, 'num_planes': args.num_planes,
        'near_plane': 1.0, 'far_plane': 100.0,
        'num_workers': 0, 'iterations_per_epoch': 100, 'save_image_every_n_steps': 100,
    }

    manifest_path = os.path.join(args.output_dir, 'training_manifest.json')
    if not os.path.exists(manifest_path):
        print(f"--- Saving training manifest to '{manifest_path}'... ---")
        # --- MODIFIED: Bake all choices into the manifest ---
        manifest_data = {
            'num_planes': config['num_planes'], 'near_plane': config['near_plane'], 'far_plane': config['far_plane'],
            'source_view_indices': config['source_view_indices'], 'ref_view_idx': config['ref_view_idx'],
            'witness_view_indices': config['witness_view_indices'],
            'resolution_w': config['resolution_w'], 'resolution_h': config['resolution_h'],
        }
        with open(manifest_path, 'w') as f: json.dump(manifest_data, f, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- RUNNING ON {device.type.upper()} ---")
    
    net = model.HybridMPIModel(num_views=len(config['source_view_indices']), num_planes=config['num_planes']).to(device)
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'])
    loss_fn = vgg.VGGPerceptualLoss(resize=False).to(device)
    
    all_scenes = sorted([d for d in os.listdir(args.scenes_path) if os.path.isdir(os.path.join(args.scenes_path, d))])
    print(f"--- Found {len(all_scenes)} scenes in '{args.scenes_path}'. ---")

    dataset = SpacesIterableDataset(scenes_base_path=args.scenes_path, scene_list=all_scenes, config=config)
    loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True)
    
    # [Rest of the training loop is unchanged]
    start_epoch, global_step = 0, 0
    if os.path.exists(config['checkpoint_path']):
        print(f"--- Found checkpoint. Resuming training. ---")
        checkpoint = torch.load(config['checkpoint_path'], map_location=device)
        net.load_state_dict(checkpoint['model_state_dict']); optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1; global_step = checkpoint.get('global_step', 0)
        print(f"--- Resuming from Epoch {start_epoch + 1}. ---")
    else: print("--- No checkpoint found. Starting from scratch. ---")

    print("\n--- Starting training loop... ---")
    data_iterator = iter(loader)
    for epoch in range(start_epoch, config['epochs']):
        progress_bar = tqdm.tqdm(range(config['iterations_per_epoch']), desc=f"Epoch {epoch+1}/{config['epochs']}")
        for i in progress_bar:
            global_step += 1; batch = next(data_iterator)
            src, poses, intr, val_imgs, val_poses, val_intr, ref_pose, ref_intr = [b.to(device) for b in batch]
            planes = torch.tensor(np.linspace(config['near_plane'], config['far_plane'], config['num_planes']), dtype=torch.float32, device=device)
            optimizer.zero_grad(); psv = utils.create_psv(src, poses, intr, ref_pose.unsqueeze(1), ref_intr.unsqueeze(1), planes)
            mpi_rgba_pred = net(psv); total_loss = 0
            for view_idx in range(len(config['witness_view_indices'])):
                tgt_pose, tgt_intr, tgt_img = val_poses[:, view_idx].unsqueeze(1), val_intr[:, view_idx].unsqueeze(1), val_imgs[:, view_idx]
                rendered = utils.render_mpi(mpi_rgba_pred, ref_pose.unsqueeze(1), ref_intr.unsqueeze(1), tgt_pose, tgt_intr, planes)
                total_loss += loss_fn(rendered.permute(0, 3, 1, 2), tgt_img)
            total_loss.backward(); optimizer.step(); loss_value = total_loss.item()
            progress_bar.set_postfix({'loss': f'{loss_value:.6f}', 'step': global_step})
            if global_step % config['save_image_every_n_steps'] == 0:
                save_snapshot_image(mpi_rgba_pred.detach(), ref_pose, ref_intr, val_imgs, val_poses, val_intr, planes, global_step, config, args.output_dir)
        torch.save({ 'epoch': epoch, 'global_step': global_step, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss_value }, config['checkpoint_path'])
    final_model_path = os.path.join(args.output_dir, 'hybrid_mpi_model.pt'); torch.save(net.state_dict(), final_model_path)
    print(f"\n--- Training complete. Final model saved to '{final_model_path}' ---")

if __name__ == '__main__':
    main()
