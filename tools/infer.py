import os
import numpy as np
import torch
import mmcv
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from opera.models import build_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PETR WiFi 姿态估计推理')
    
    # 必需的参数
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型检查点路径')
    parser.add_argument('data_root_path', help='数据根目录')
    parser.add_argument('file_name', help='文件名')
    
    # 可选参数
    parser.add_argument('--vis-dir', default='visualization_results',
                        help='可视化结果保存目录 (默认: visualization_results)')
    parser.add_argument('--device', default='cuda:0',
                        help='设备 (默认: cuda:0)')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='置信度阈值 (默认: 0.3)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批处理大小 (默认: 1)')
    parser.add_argument('--save-data', action='store_true',
                        help='是否保存处理后的数据')
    
    args = parser.parse_args()
    return args

def load_wifi_data(data_path):
    """加载并处理 WiFi 数据"""
    print(f"正在加载 WiFi 数据: {data_path}")
    
    if data_path.endswith('.mat'):
        import h5py
        import pywt
        
        # 加载数据
        with h5py.File(data_path, 'r') as f:
            csi = f['csi_out']
            csi = csi['real'][:] + 1j * csi['imag'][:]
            
        # 转换维度顺序
        csi = np.array(csi).transpose(3, 2, 1, 0)
        csi = csi.astype(np.complex128)
        
        # 使用小波变换处理幅度
        def dwt_amp(csi):
            w = pywt.Wavelet('dB11')
            list = pywt.wavedec(abs(csi), w, 'sym')
            csi_amp = pywt.waverec(list, w)
            return csi_amp
        
        # 相位去噪
        def CSI_sanitization(csi_rx):
            one_csi = csi_rx[0,:,:]
            two_csi = csi_rx[1,:,:]
            three_csi = csi_rx[2,:,:]
            pi = np.pi
            M = 3
            N = 30
            T = one_csi.shape[1]
            fi = 312.5 * 2
            
            csi_phase = np.zeros((M, N, T))
            for t in range(T):
                csi_phase[0, :, t] = np.unwrap(np.angle(one_csi[:, t]))
                csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(two_csi[:, t] * np.conj(one_csi[:, t])))
                csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(three_csi[:, t] * np.conj(two_csi[:, t])))
                
                ai = np.tile(2 * pi * fi * np.array(range(N)), M)
                bi = np.ones(M * N)
                ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t]))
                
                A = np.dot(ai, ai)
                B = np.dot(ai, bi)
                C = np.dot(bi, bi)
                D = np.dot(ai, ci)
                E = np.dot(bi, ci)
                
                rho_opt = (B * E - C * D) / (A * C - B ** 2)
                beta_opt = (B * D - A * E) / (A * C - B ** 2)
                
                temp = np.tile(np.array(range(N)), M).reshape(M, N)
                csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
                
            antennaPair_One = abs(one_csi) * np.exp(1j * csi_phase[0, :, :])
            antennaPair_Two = abs(two_csi) * np.exp(1j * csi_phase[1, :, :])
            antennaPair_Three = abs(three_csi) * np.exp(1j * csi_phase[2, :, :])
            
            return np.stack([antennaPair_One, antennaPair_Two, antennaPair_Three])
        
        def phase_deno(csi):
            ph_rx1 = CSI_sanitization(csi[0,:,:,:])
            ph_rx2 = CSI_sanitization(csi[1,:,:,:])
            ph_rx3 = CSI_sanitization(csi[2,:,:,:])
            return np.stack([ph_rx1, ph_rx2, ph_rx3])
        
        # 处理幅度和相位
        csi_amp = dwt_amp(csi)
        csi_ph = phase_deno(csi)
        csi_ph = np.angle(csi_ph)
        
        # 合并幅度和相位信息
        csi = np.concatenate((csi_amp, csi_ph), axis=2)
        
        # 转换为 tensor 并调整维度顺序
        # [3, 3, 20, 60] -> [1, 3, 3, 20, 60]
        csi = torch.FloatTensor(csi).permute(0, 1, 3, 2)
        csi = csi.unsqueeze(0)  # 添加批次维度
        
        print(f"处理后数据形状: {csi.shape}")
        
        return csi
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")

def init_model(config, checkpoint=None, device='cuda:0'):
    """初始化 PETR 模型"""
    config = mmcv.Config.fromfile(config)
    model = build_model(config.model)
    if checkpoint:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device)
    model.eval()
    return model

def visualize_3d_pose(pred_keypoints, gt_keypoints=None, score_thr=0.3, save_path=None):
    """使用 plotly 进行交互式 3D 人体姿态可视化，支持对比显示"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # 转换数据类型
    if isinstance(pred_keypoints, (list, tuple)):
        pred_keypoints = np.array(pred_keypoints)
    if isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = pred_keypoints.cpu().numpy()
    
    if gt_keypoints is not None:
        if isinstance(gt_keypoints, (list, tuple)):
            gt_keypoints = np.array(gt_keypoints)
        if isinstance(gt_keypoints, torch.Tensor):
            gt_keypoints = gt_keypoints.cpu().numpy()
    
    print(f"预测关键点:")
    print(f"- 形状: {pred_keypoints.shape}")
    print(f"- 类型: {type(pred_keypoints)}")
    print(f"- 数值范围: [{pred_keypoints.min():.3f}, {pred_keypoints.max():.3f}]")
    
    # 定义人体骨架连接，参考 WifiPoseDataset 的实现
    bones = [
        [0, 1],   # 0 -> 1
        [1, 2],   # 1 -> 2
        [2, 5],   # 2 -> 5
        [3, 0],   # 3 -> 0
        [4, 2],   # 4 -> 2
        [5, 7],   # 5 -> 7
        [6, 3],   # 6 -> 3
        [7, 3],   # 7 -> 3
        [8, 4],   # 8 -> 4
        [9, 5],   # 9 -> 5
        [10, 6],  # 10 -> 6
        [11, 7],  # 11 -> 7
        [12, 9],  # 12 -> 9
        [13, 11]  # 13 -> 11
    ]
    
    # 创建两个子图
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Predicted Pose', 'Ground Truth vs Predicted')
    )
    
    # 定义颜色
    pred_colors = ['blue', 'red', 'green', 'cyan', 'magenta']
    gt_color = 'gray'  # 真实值使用灰色
    
    # 左图：只显示预测结果
    for person_idx, kpts in enumerate(pred_keypoints):
        color = pred_colors[person_idx % len(pred_colors)]
        
        # 绘制骨架
        for start_idx, end_idx in bones:
            if start_idx < len(kpts) and end_idx < len(kpts):
                fig.add_trace(
                    go.Scatter3d(
                        x=[kpts[start_idx, 0], kpts[end_idx, 0]],
                        y=[kpts[start_idx, 1], kpts[end_idx, 1]],
                        z=[kpts[start_idx, 2], kpts[end_idx, 2]],
                        mode='lines',
                        line=dict(color=color, width=3),
                        name=f'Pred Person {person_idx+1}'
                    ),
                    row=1, col=1
                )
        
        # 绘制关键点
        fig.add_trace(
            go.Scatter3d(
                x=kpts[:, 0],
                y=kpts[:, 1],
                z=kpts[:, 2],
                mode='markers',
                marker=dict(size=5, color=color),
                name=f'Pred Joints {person_idx+1}'
            ),
            row=1, col=1
        )
    
    # 右图：对比显示
    if gt_keypoints is not None:
        # 绘制真实值
        for person_idx, kpts in enumerate(gt_keypoints):
            # 绘制骨架
            for start_idx, end_idx in bones:
                if start_idx < len(kpts) and end_idx < len(kpts):
                    fig.add_trace(
                        go.Scatter3d(
                            x=[kpts[start_idx, 0], kpts[end_idx, 0]],
                            y=[kpts[start_idx, 1], kpts[end_idx, 1]],
                            z=[kpts[start_idx, 2], kpts[end_idx, 2]],
                            mode='lines',
                            line=dict(color=gt_color, width=3),
                            name=f'GT Person {person_idx+1}'
                        ),
                        row=1, col=2
                    )
            
            # 绘制关键点
            fig.add_trace(
                go.Scatter3d(
                    x=kpts[:, 0],
                    y=kpts[:, 1],
                    z=kpts[:, 2],
                    mode='markers',
                    marker=dict(size=5, color=gt_color),
                    name=f'GT Joints {person_idx+1}'
                ),
                row=1, col=2
            )
        
        # 在右图中也显示预测值
        for person_idx, kpts in enumerate(pred_keypoints):
            color = pred_colors[person_idx % len(pred_colors)]
            
            # 绘制骨架
            for start_idx, end_idx in bones:
                if start_idx < len(kpts) and end_idx < len(kpts):
                    fig.add_trace(
                        go.Scatter3d(
                            x=[kpts[start_idx, 0], kpts[end_idx, 0]],
                            y=[kpts[start_idx, 1], kpts[end_idx, 1]],
                            z=[kpts[start_idx, 2], kpts[end_idx, 2]],
                            mode='lines',
                            line=dict(color=color, width=3),
                            name=f'Pred Person {person_idx+1}'
                        ),
                        row=1, col=2
                    )
            
            # 绘制关键点
            fig.add_trace(
                go.Scatter3d(
                    x=kpts[:, 0],
                    y=kpts[:, 1],
                    z=kpts[:, 2],
                    mode='markers',
                    marker=dict(size=5, color=color),
                    name=f'Pred Joints {person_idx+1}'
                ),
                row=1, col=2
            )
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text='3D Pose Estimation Comparison',
            x=0.5,
            y=0.95
        ),
        showlegend=True,
        width=1600,
        height=800,
        scene1=dict(
            xaxis=dict(title='X', showgrid=True, zeroline=True),
            yaxis=dict(title='Y', showgrid=True, zeroline=True),
            zaxis=dict(title='Z', showgrid=True, zeroline=True, autorange='reversed'),  # 反转 Z 轴
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube'
        ),
        scene2=dict(
            xaxis=dict(title='X', showgrid=True, zeroline=True),
            yaxis=dict(title='Y', showgrid=True, zeroline=True),
            zaxis=dict(title='Z', showgrid=True, zeroline=True, autorange='reversed'),  # 反转 Z 轴
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    if save_path:
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['hoverclosest', 'hovercompare'],
            'scrollZoom': True
        }
        fig.write_html(save_path, include_plotlyjs=True, full_html=True, config=config)
        print(f"已保存交互式图像到: {save_path}")
    else:
        fig.show(renderer="browser")

def process_wifi_data(model, wifi_data, score_thr=0.3, device='cuda:0'):
    """处理 WiFi 数据并返回检测结果"""
    # 准备数据
    data = dict(
        img=wifi_data,  # [B, C, H, W, F]
        img_metas=[[{
            'filename': None,
            'ori_filename': None,
            'ori_shape': wifi_data.shape,
            'img_shape': wifi_data.shape,
            'pad_shape': wifi_data.shape,
            'scale_factor': 1.0,
            'flip': False,
            'flip_direction': None,
        }]],
        gt_labels=np.zeros(1, dtype=np.int64),
        gt_bboxes=torch.zeros((0, 4)),
        gt_areas=torch.zeros(0),
        gt_keypoints=torch.zeros((1, 14, 3))
    )

    # 转到GPU
    if device != 'cpu':
        if isinstance(wifi_data, torch.Tensor):
            wifi_data = wifi_data.to(device)
        data['img'] = [wifi_data]
        
    # 模型推理
    with torch.no_grad():
        results = model.simple_test(
            data['img'][0],
            data['img_metas'][0],
            rescale=True
        )
    
    return results

def visualize_results(wifi_data, keypoints, score_thr=0.3, save_path=None):
    """可视化 3D 姿态估计结果"""
    import plotly.io as pio
    # 设置默认渲染器
    pio.renderers.default = "browser"
    
    # 创建图形
    fig = go.Figure()
    
    # 转换数据类型
    if isinstance(keypoints, (list, tuple)):
        keypoints = np.array(keypoints)
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    
    print(f"关键点数据形状: {keypoints.shape}")
    
    # 定义人体骨架连接
    bones = [
        [0, 1],   # 0 -> 1
        [1, 2],   # 1 -> 2
        [2, 5],   # 2 -> 5
        [3, 0],   # 3 -> 0
        [4, 2],   # 4 -> 2
        [5, 7],   # 5 -> 7
        [6, 3],   # 6 -> 3
        [7, 3],   # 7 -> 3
        [8, 4],   # 8 -> 4
        [9, 5],   # 9 -> 5
        [10, 6],  # 10 -> 6
        [11, 7],  # 11 -> 7
        [12, 9],  # 12 -> 9
        [13, 11]  # 13 -> 11
    ]
    
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow']
    
    # 对每个检测到的人进行绘制
    for person_idx, kpts in enumerate(keypoints):
        color = colors[person_idx % len(colors)]
        
        # 绘制骨架
        for start_idx, end_idx in bones:
            if start_idx < len(kpts) and end_idx < len(kpts):
                fig.add_trace(
                    go.Scatter3d(
                        x=[kpts[start_idx, 0], kpts[end_idx, 0]],
                        y=[kpts[start_idx, 1], kpts[end_idx, 1]],
                        z=[kpts[start_idx, 2], kpts[end_idx, 2]],
                        mode='lines',
                        line=dict(color=color, width=3),
                        name=f'Person {person_idx+1} Bone'
                    )
                )
        
        # 绘制关键点
        fig.add_trace(
            go.Scatter3d(
                x=kpts[:, 0],
                y=kpts[:, 1],
                z=kpts[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                ),
                name=f'Person {person_idx+1} Joints'
            )
        )
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text='3D Pose Estimation',
            x=0.5,
            y=0.95
        ),
        showlegend=True,
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(title='X', showgrid=True, zeroline=True),
            yaxis=dict(title='Y', showgrid=True, zeroline=True),
            zaxis=dict(title='Z', showgrid=True, zeroline=True),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=50, b=0)  # 减小边距
    )
    
    if save_path:
        # 保存为独立的 HTML 文件，确保包含所有必要的依赖
        config = {
            'displayModeBar': True,  # 显示工具栏
            'displaylogo': False,    # 不显示 plotly logo
            'modeBarButtonsToAdd': ['hoverclosest', 'hovercompare'],  # 添加额外的工具按钮
            'scrollZoom': True       # 启用滚轮缩放
        }
        
        # 使用 full_html=True 确保包含所有依赖
        fig.write_html(
            save_path,
            include_plotlyjs=True,
            full_html=True,
            include_mathjax=False,
            config=config
        )
        print(f"已保存交互式图像到: {save_path}")
    else:
        # 在浏览器中显示
        fig.show(renderer="browser", config={
            'displayModeBar': True,
            'displaylogo': False,
            'scrollZoom': True
        })

def inference_wifi(model, wifi_data, gt_data=None, score_thr=0.3, device='cuda:0', vis_dir=None):
    """使用 PETR 模型进行 WiFi 姿态估计推理并可视化"""
    results = process_wifi_data(model, wifi_data, score_thr, device)
    
    for frame_idx, result in enumerate(results):
        det_bboxes, det_kpts = result
        
        if isinstance(det_bboxes, (list, tuple)):
            det_bboxes = np.array(det_bboxes)
        if isinstance(det_kpts, (list, tuple)):
            det_kpts = np.array(det_kpts)
            
        print(f"\n处理第 {frame_idx} 帧:")
        print(f"检测框形状: {det_bboxes.shape}")
        print(f"关键点形状: {det_kpts.shape}")
        
        for batch_idx in range(det_bboxes.shape[0]):
            batch_bboxes = det_bboxes[batch_idx]
            batch_kpts = det_kpts[batch_idx]
            
            scores = batch_bboxes[:, -1]
            high_conf = scores > score_thr
            filtered_kpts = batch_kpts[high_conf]
            
            if len(filtered_kpts) > 0:
                if vis_dir:
                    save_path = f'{vis_dir}/frame_{frame_idx}_batch_{batch_idx}.html'
                else:
                    save_path = None
                
                print(f"检测到 {len(filtered_kpts)} 个高置信度的姿态")
                
                # 获取对应的真实值（如果有）
                current_gt = None
                if gt_data is not None:
                    current_gt = gt_data  # 直接使用加载的真实值数据
                
                visualize_3d_pose(filtered_kpts, current_gt, score_thr=score_thr, save_path=save_path)
    
    return results

def load_gt_data(gt_path):
    """加载真实值关键点数据，参考 WifiPoseDataset 实现
    
    Args:
        gt_path (str): 真实值数据文件路径，支持 .npy 格式
        
    Returns:
        np.ndarray: 关键点数据，形状为 [num_persons, num_joints, 3]
    """
    print(f"正在加载真实值数据: {gt_path}")
    
    if gt_path.endswith('.npy'):
        try:
            # 直接加载 .npy 文件
            gt_data = np.load(gt_path)
            gt_data = np.array(gt_data, dtype=np.float32)
            
            # 检查维度
            if len(gt_data.shape) == 3:  # [num_persons, num_joints, 3]
                print(f"加载完成！数据形状: {gt_data.shape}")
                print(f"数值范围: [{gt_data.min():.3f}, {gt_data.max():.3f}]")
                return gt_data
            else:
                raise ValueError(f"不支持的数据形状: {gt_data.shape}, 期望形状: [num_persons, num_joints, 3]")
            
        except Exception as e:
            print(f"加载 .npy 文件失败: {str(e)}")
            raise
    else:
        raise ValueError(f"不支持的文件格式: {gt_path}, 请使用 .npy 格式")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建可视化目录
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"创建可视化目录: {args.vis_dir}")
    
    try:
        # 初始化模型
        print(f"正在加载模型配置: {args.config}")
        print(f"正在加载模型检查点: {args.checkpoint}")
        model = init_model(args.config, args.checkpoint, device=args.device)
        
        # 加载数据
        data_path = os.path.join(args.data_root_path, "csi", args.file_name + ".mat")
        wifi_data = load_wifi_data(data_path)
        
        # 加载真实值数据（如果有）
        gt_data = None
        gt_path = os.path.join(args.data_root_path, "keypoint", args.file_name + ".npy")
        if gt_path:  # 需要在 parse_args 中添加 gt_path 参数
            gt_data = load_gt_data(gt_path)  # 需要实现 load_gt_data 函数
        
        # 推理并可视化
        print(f"开始推理，置信度阈值: {args.score_thr}")
        results = inference_wifi(
            model,
            wifi_data,
            gt_data,
            score_thr=args.score_thr,
            device=args.device,
            vis_dir=args.vis_dir
        )
        
        print(f'处理完成！可视化结果已保存到: {args.vis_dir}')
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()