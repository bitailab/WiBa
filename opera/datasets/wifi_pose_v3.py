import os
from scipy import io
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import pywt
from collections import OrderedDict
import scipy.fft as fft
from .builder import DATASETS
from mmdet.datasets.pipelines import Compose
import h5py
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm
import time
import psutil
from multiprocessing import Pool

@DATASETS.register_module()
class WifiPoseDataset(dataset):
    CLASSES = ('person', )
    def __init__(self, dataset_root, pipeline, mode, load_settings=None, **kwargs):
        super().__init__()
        
        # 打印接收到的所有参数
        print("Dataset initialization parameters:")
        print(f"dataset_root: {dataset_root}")
        print(f"mode: {mode}")
        print(f"load_settings: {load_settings}")
        print(f"kwargs: {kwargs}")
        
        self.data_root = dataset_root
        self.pipeline = Compose(pipeline)
        self.filename_list = self.load_file_name_list(os.path.join(self.data_root, mode + '_data_list.txt'))
        self._set_group_flag()
        
        # 获取加载设置，使用默认值如果未指定
        self.load_settings = load_settings or {}
        self.max_workers = self.load_settings.get('max_workers', 8)
        
        # 使用多个缓存锁来减少锁竞争
        self.data_cache = {}
        self.cache_locks = [threading.Lock() for _ in range(10)]  # 创建多个锁
        
        # 使用多线程预加载数据
        self._preload_data_parallel()
        
    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir

    def CSI_sanitization(self, csi_rx):
        """CSI相位清洗"""
        N_sc = csi_rx.shape[1]  # 子载波数量
        N_t = csi_rx.shape[0]   # 发射天线数量
        
        # 计算相位差
        phi_diff = np.zeros((N_t-1, N_sc), dtype=np.complex128)
        for i in range(N_t-1):
            phi_diff[i,:] = np.multiply(csi_rx[i,:], np.conj(csi_rx[i+1,:]))
        
        # 解包裹相位
        for i in range(N_t-1):
            phi_diff[i,:] = np.unwrap(np.angle(phi_diff[i,:]))
            
        # 相位清洗
        clean_phi_diff = np.zeros_like(phi_diff)
        for i in range(N_t-1):
            # 使用FFT进行相位清洗
            fft_diff = fft.fft(phi_diff[i,:])
            # 确保高频分量置零的索引不超出数组范围
            cutoff = min(20, fft_diff.shape[0]//2)
            fft_diff[cutoff:-cutoff] = 0 if cutoff < fft_diff.shape[0]//2 else 0
            clean_phi_diff[i,:] = fft.ifft(fft_diff).real
            
        # 重构相位
        clean_phase = np.zeros((N_t, N_sc), dtype=np.complex128)
        clean_phase[0,:] = csi_rx[0,:]
        for i in range(N_t-1):
            # 确保形状匹配
            phase_exp = np.exp(1j * clean_phi_diff[i,:])
            clean_phase[i+1,:] = clean_phase[i,:] * phase_exp
            
        return clean_phase
        
    def phase_deno(self, csi):
        """相位去噪"""
        results = []
        for i in range(csi.shape[0]):
            # 确保输入数据维度正确
            rx_data = csi[i,:,:,:]
            if rx_data.ndim == 3:  # 如果是3维数据
                sanitized = self.CSI_sanitization(rx_data.reshape(rx_data.shape[0], -1))
                # 恢复原始形状
                sanitized = sanitized.reshape(rx_data.shape)
            else:
                sanitized = self.CSI_sanitization(rx_data)
            results.append(sanitized)
        return np.stack(results)
        
    def _get_cache_lock(self, idx):
        """获取对应的缓存锁"""
        return self.cache_locks[idx % len(self.cache_locks)]
        
    def _process_single_item(self, idx):
        try:
            data_name = self.filename_list[idx]
            
            # 加载CSI数据
            csi_path = os.path.join(self.data_root, 'csi', f'{data_name}.mat')
            csi = h5py.File(csi_path)['csi_out'].value
            csi = csi['real'] + csi['imag']*1j
            csi = np.array(csi).transpose(3,2,1,0)
            csi = csi.astype(np.complex128)
            
            # 预处理CSI数据
            csi_amp = self.dwt_amp(csi)
            csi_ph = self.phase_deno(csi)
            csi_ph = np.angle(csi_ph)
            
            # 合并并转换格式
            csi_processed = np.concatenate((csi_amp, csi_ph), axis=2)
            csi_processed = torch.FloatTensor(csi_processed).permute(0,1,3,2)
            
            # 加载和处理关键点数据
            keypoint_path = os.path.join(self.data_root, 'keypoint', f'{data_name}.npy')
            keypoint = np.array(np.load(keypoint_path))
            keypoint = torch.FloatTensor(keypoint)
            
            # 准备其他数据
            numOfPerson = keypoint.shape[0]
            gt_labels = np.zeros(numOfPerson, dtype=np.int64)
            gt_bboxes = torch.tensor([])
            gt_areas = torch.tensor([])
            
            processed_data = {
                'img': csi_processed,
                'gt_keypoints': keypoint,
                'gt_labels': gt_labels,
                'gt_bboxes': gt_bboxes,
                'gt_areas': gt_areas,
                'img_name': data_name
            }
            
            # 使用分段锁而不是全局锁
            with self._get_cache_lock(idx):
                self.data_cache[idx] = processed_data
                
        except Exception as e:
            print(f"处理数据 {idx} 时出错: {str(e)}")
            
    def _preload_data_parallel(self):
        """并行预加载所有数据"""
        print(f"开始并行加载和预处理数据... (使用 {self.max_workers} 个线程)")
        total = len(self.filename_list)
        
        # 创建进度条
        pbar = tqdm(total=total, desc="数据加载进度")
        
        def update_progress(*args):
            pbar.update(1)
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for idx in range(total):
                future = executor.submit(self._process_single_item, idx)
                future.add_done_callback(update_progress)
                futures.append(future)
            
            # 等待所有任务完成
            for future in futures:
                future.result()
        
        pbar.close()
        print("数据预加载和预处理完成！")
        
    def get_item_single_frame(self, index):
        """直接返回预处理好的数据"""
        return self.data_cache[index]

    def __getitem__(self, index):
        """直接使用预处理好的数据"""
        result = self.get_item_single_frame(index)
        return self.pipeline(result)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  
                if not lines:
                    break
                file_name_list.append(lines.split()[0])
        return file_name_list

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    @staticmethod
    def dwt_amp(csi):
        """优化的dwt_amp实现"""
        w = pywt.Wavelet('dB11')
        results = []
        for channel in csi:
            list = pywt.wavedec(abs(channel), w, 'sym')
            results.append(pywt.waverec(list, w))
        return np.array(results)

    def keypoint_process(self, keypoints):
        next_point = np.array([[0,1], [1,2], [2,5], [3,0], [4,2], [5,7],
                               [6,3], [7,3], [8,4], [9,5], [10,6], [11,7],
                               [12,9], [13,11]])
        keypoints_list = []
        for numofperson in range(keypoints.shape[0]):
            for numofpoint in range(keypoints.shape[1]):
                point_with_next = np.concatenate((keypoints[numofperson,next_point[numofpoint,0],:],
                                                  keypoints[numofperson,next_point[numofpoint,1],:]), axis=0)
                point_class = np.zeros((15))
                keypoints_list.append(point_with_next)
        
        return np.array(keypoints_list)
    
    def evaluate(self,
                 results,
                 metric='keypoints',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        mpjpe_3d_list = []
        mpjpe_h_list = []
        mpjpe_v_list = []
        mpjpe_d_list = []
        for i in range(len(results)):
            info = self.get_item_single_frame(i)
            gt_keypoints = info['gt_keypoints']
            data_name = info['img_name']
            det_bboxes, det_keypoints = results[i]
            for label in range(len(det_keypoints)):
                kpt_pred = det_keypoints[label]
                kpt_pred = torch.tensor(kpt_pred, dtype=gt_keypoints.dtype, device=gt_keypoints.device)
                #np.save('/home/yankangwei/opera-main/result/pose_o/%s.npy' %data_name, kpt_pred)
                mpjpe_3d,mpjpeh,mpjpev,mpjped = self.calc_mpjpe(gt_keypoints, kpt_pred, data_name, root = [5,7])
                mpjpe_3d_list.append(mpjpe_3d.numpy())
                mpjpe_h_list.append(mpjpeh.numpy())
                mpjpe_v_list.append(mpjpev.numpy())
                mpjpe_d_list.append(mpjped.numpy())
                #mpjpe_3d_list.append(np.array([0]))

        mpjpe = np.array(mpjpe_3d_list).mean()   
        mpjpeh = np.array(mpjpe_h_list).mean() 
        mpjpev = np.array(mpjpe_v_list).mean() 
        mpjped = np.array(mpjpe_d_list).mean() 
        result = {'mpjpe':mpjpe, 'mpjpeh':mpjpeh, 'mpjpev':mpjpev, 'mpjped':mpjped}
        return OrderedDict(result)
    
    def calc_mpjpe(self, real, pred, no, root=0):
        n = real.shape[0]
        m = pred.shape[0]
        j, c = pred.shape[1:]
        assert j == real.shape[1] and c == real.shape[2]
        if isinstance(root,list):
            real_root = real.unsqueeze(1).expand(n, m, j, c)
            pred_root = pred.unsqueeze(0).expand(n, m, j, c)
            #n*m*j  n*j
            distance_array = torch.ones((n,m), dtype=torch.float) * 2 ** 24  # TODO: magic number!
            for i in range(n):
                for j in range(m):
                    distance_array[i][j] = torch.norm(real[i]-pred[j], p=2, dim=-1).mean()

            # distance_array = torch.norm(real_root-pred_root, p=2, dim=-1)*vis_mask.unsqueeze(1).expand(n, m, j)
            # distance_array = distance_array.sum(-1) / vis_mask.sum(-1).unsqueeze(1)
            # print(torch.min(distance_array))
        else:
            real_root = real[:, root].unsqueeze(0).expand(n, m, c)
            pred_root = pred[:, root].unsqueeze(1).expand(n, m, c)
            distance_array = torch.pow(real_root - pred_root, 2)
        corres = torch.ones(n, dtype=torch.long)*-1
        occupied = torch.zeros(m, dtype=torch.long)

        while torch.min(distance_array) < 50:   # threshold 30.
            min_idx = torch.where(distance_array == torch.min(distance_array))
            
            for i in range(len(min_idx[0])):
                distance_array[min_idx[0][i]][min_idx[1][i]] = 50
                if corres[min_idx[0][i]] >= 0 or occupied[min_idx[1][i]]:
                    continue
                else:
                    corres[min_idx[0][i]] = min_idx[1][i]
                    occupied[min_idx[1][i]] = 1
        new_pred = pred[corres]
        #np.save('/data/repos/Person-in-WiFi-3D-repo/result/pose_pred/%s.npy' %no, new_pred)
        #np.save('/data/repos/Person-in-WiFi-3D-repo/result/pose_gt/%s.npy' %no, real)
        mpjpe = torch.sqrt(torch.pow(real - new_pred, 2).sum(-1))
        mpjpeh = torch.sqrt(torch.pow(real[:,:,0] - new_pred[:,:,0], 2))
        mpjpev = torch.sqrt(torch.pow(real[:,:,1] - new_pred[:,:,1], 2))
        mpjped = torch.sqrt(torch.pow(real[:,:,2] - new_pred[:,:,2], 2))
        # mpjpe = torch.norm(real-new_pred, p=2, dim=-1) #n*j
        # mpjpe_mean = (mpjpe*vis_mask.float()).sum(-1)/vis_mask.float().sum(-1) if vis_mask is not None else mpjpe.mean(-1)
        return mpjpe.mean()*1000, mpjpeh.mean()*1000, mpjpev.mean()*1000, mpjped.mean()*1000
# if __name__ == "__main__":
#     path = 'data/'
#     train_ds = Train_Dataset(path)
#     train_dl = DataLoader(train_ds, 1, False, num_workers=1)
