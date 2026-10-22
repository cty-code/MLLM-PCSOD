import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset

class PCSOD(Dataset):
    def __init__(self, split='train', data_root='./data/', num_point=4096, transform=None):
        super(PCSOD, self).__init__()
        self.split = split
        self.num_point = num_point
        self.transform = transform
        self.data_root = data_root

        self.scene_coord_min = []
        self.scene_coord_max = []
        self.scenes_split = []
        self.num_point_all = []

        labelweights = np.zeros(2)

        print('Computing weights for cross-entropy loss...')
        for split_dir in ['train', 'test']:
            scene_names = os.listdir(os.path.join(self.data_root, split_dir, 'gt'))
            for scene_name in tqdm(scene_names, total=len(scene_names)):
                scene_label_path = os.path.join(self.data_root, split_dir, 'gt', scene_name)
                scene_data = o3d.io.read_point_cloud(scene_label_path)
                pts = np.asarray(scene_data.points)
                lbls = np.asarray(scene_data.colors)[..., 0]
                lbls[lbls != 0] = 1
                tmp, _ = np.histogram(lbls, range(3))
                labelweights += tmp

                if self.split == split_dir:
                    self.scenes_split.append(scene_name)
                    coord_min = np.amin(pts, axis=0)
                    coord_max = np.amax(pts, axis=0)
                    self.scene_coord_min.append(coord_min)
                    self.scene_coord_max.append(coord_max)
                    self.num_point_all.append(lbls.size)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1/3.0)

        print(f"Totally {len(self.scenes_split)} samples in {self.split} set.")

    def __getitem__(self, item):
        scene_name = self.scenes_split[item]

        scene_point_path = os.path.join(self.data_root, self.split, "point", scene_name)
        scene_label_path = os.path.join(self.data_root, self.split, "gt", scene_name)

        pcd = o3d.io.read_point_cloud(scene_point_path)
        full_pts = np.concatenate([
            np.asarray(pcd.points, np.float32),
            np.asarray(pcd.colors, np.float32)
        ], axis=1)  

        gt_pcd = o3d.io.read_point_cloud(scene_label_path)
        full_lbl = np.asarray(gt_pcd.colors, np.float32)[..., 0]
        full_lbl[full_lbl != 0] = 1
        if self.split == 'train':
            scale = np.random.uniform(0.9, 1.1)
            full_pts[:, 0:3] *= scale
            
            angle = np.random.uniform(0, 2 * np.pi)
            cosval, sinval = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cosval, -sinval, 0],
                                   [sinval,  cosval, 0],
                                   [     0,       0, 1]], dtype=np.float32)
            full_pts[:, 0:3] = np.dot(full_pts[:, 0:3], rot_matrix)
            
            shift = np.random.uniform(-0.1, 0.1, 3)
            full_pts[:, 0:3] += shift

            coord_min = np.amin(full_pts[:, 0:3], axis=0)
            coord_max = np.amax(full_pts[:, 0:3], axis=0)
        else:
            coord_min = self.scene_coord_min[item]
            coord_max = self.scene_coord_max[item]

        N_all = full_pts.shape[0]

        idx = np.arange(N_all)
        if self.split == "train":
            np.random.shuffle(idx)
        
        sampled_idx = idx[:self.num_point] if N_all >= self.num_point else idx

        local_pts6 = full_pts[sampled_idx]     
        local_lbls = full_lbl[sampled_idx]     

        nloc = local_pts6.shape[0]
        
        if self.split == 'train':
            noise = np.clip(np.random.normal(0, 0.001, (nloc, 3)), -0.005, 0.005)
            local_pts6[:, 0:3] += noise

        output_pts9 = np.zeros((self.num_point, 9), dtype=np.float32)
        output_lbls = np.zeros((self.num_point,), dtype=np.int64)
        
        output_pts9[:nloc, 0:6] = local_pts6
        output_lbls[:nloc] = local_lbls

        output_pts9[:nloc, 6] = (local_pts6[:, 0] - coord_min[0]) / (coord_max[0] - coord_min[0] + 1e-6)
        output_pts9[:nloc, 7] = (local_pts6[:, 1] - coord_min[1]) / (coord_max[1] - coord_min[1] + 1e-6)
        output_pts9[:nloc, 8] = (local_pts6[:, 2] - coord_min[2]) / (coord_max[2] - coord_min[2] + 1e-6)

        output_pts9[:nloc, 0] = local_pts6[:, 0] - coord_max[0] / 2
        output_pts9[:nloc, 1] = local_pts6[:, 1] - coord_max[1] / 2

        if self.transform is not None:
            output_pts9, output_lbls = self.transform(output_pts9, output_lbls)

        return (
            output_pts9,      
            output_lbls,       
            full_pts,          
            full_lbl,        
            coord_min.astype(np.float32),  
            coord_max.astype(np.float32),
            scene_name
        )

    def __len__(self):
        return len(self.scenes_split)

if __name__ == '__main__':
    data, label, f_p, f_l, c_min, c_max, s_n = next(iter(PCSOD()))
    print(f"Output shape: {data.shape}")  # 应该打印 (4096, 9)
    print(f"Label shape: {label.shape}")
