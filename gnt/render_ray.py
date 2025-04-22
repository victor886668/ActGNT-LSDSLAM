import torch
from collections import OrderedDict

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################


def sample_pdf(bins, weights, N_samples, det=False):
    """
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    """

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)  # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)  # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)  # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i : i + 1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)  # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]  # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1] - bins_g[:, :, 0])

    return samples


def sample_along_camera_ray(ray_o, ray_d, depth_range, N_samples, inv_uniform=False, det=False):
    """
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    """
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0, 0]
    far_depth_value = depth_range[0, 1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])
    if inv_uniform:
        start = 1.0 / near_depth  # [N_rays,]
        step = (1.0 / far_depth - start) / (N_samples - 1)
        inv_z_vals = torch.stack(
            [start + i * step for i in range(N_samples)], dim=1
        )  # [N_rays, N_samples]
        z_vals = 1.0 / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples - 1)
        z_vals = torch.stack(
            [start + i * step for i in range(N_samples)], dim=1
        )  # [N_rays, N_samples]

    if not det:
        # get intervals between samples
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o  # [N_rays, N_samples, 3]
    return pts, z_vals


########################################################################################################################
# ray rendering of nerf
########################################################################################################################


def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    """
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    """
    rgb = raw[:, :, :3]  # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]  # [N_rays, N_samples]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    sigma2alpha = lambda sigma, dists: 1.0 - torch.exp(-sigma)

    # point samples are ordered with increasing depth
    # interval between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

    # Eq. (3): T
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)[:, :-1]  # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weights along a ray, are always inside [0, 1]
    weights = alpha * T  # [N_rays, N_samples]
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - torch.sum(weights, dim=-1, keepdim=True))

    mask = (
        mask.float().sum(dim=1) > 8
    )  # should at least have 8 valid observation on the ray, otherwise don't consider its loss
    depth_map = torch.sum(weights * z_vals, dim=-1)  # [N_rays,]

    ret = OrderedDict(
        [
            ("rgb", rgb_map),
            ("depth", depth_map),
            ("weights", weights),  # used for importance sampling of fine samples
            ("mask", mask),
            ("alpha", alpha),
            ("z_vals", z_vals),
        ]
    )

    return ret


def sample_fine_pts(inv_uniform, N_importance, det, N_samples, ray_batch, weights, z_vals):
    if inv_uniform:
        inv_z_vals = 1.0 / z_vals
        inv_z_vals_mid = 0.5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])  # [N_rays, N_samples-1]
        weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
        inv_z_vals = sample_pdf(
            bins=torch.flip(inv_z_vals_mid, dims=[1]),
            weights=torch.flip(weights, dims=[1]),
            N_samples=N_importance,
            det=det,
        )  # [N_rays, N_importance]
        z_samples = 1.0 / inv_z_vals
    else:
        # take mid-points of depth samples
        z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])  # [N_rays, N_samples-1]
        weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
        z_samples = sample_pdf(
            bins=z_vals_mid, weights=weights, N_samples=N_importance, det=det
        )  # [N_rays, N_importance]

    z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]

    # samples are sorted with increasing depth
    z_vals, _ = torch.sort(z_vals, dim=-1)
    N_total_samples = N_samples + N_importance

    viewdirs = ray_batch["ray_d"].unsqueeze(1).repeat(1, N_total_samples, 1)
    ray_o = ray_batch["ray_o"].unsqueeze(1).repeat(1, N_total_samples, 1)
    pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]
    return pts, z_vals


def choose_new_ks():
    for train_data in train_loader:
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Start of core optimization loop

        # load training rays
        ray_sampler = RaySamplerSingleImage(train_data, device)
        N_rand = int(
            1.0 * args.N_rand * args.num_source_views / train_data["src_rgbs"][0].shape[0]
        )
        ray_batch = ray_sampler.random_sample(
            N_rand,
            sample_mode=args.sample_mode,
            center_ratio=args.center_ratio,
        )

        featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))

        ret = render_rays(
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            featmaps=featmaps,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            N_importance=args.N_importance,
            det=args.det,
            white_bkgd=args.white_bkgd,
            ret_alpha=args.N_importance > 0,
            single_net=args.single_net,
        )

def choose_new_k(H, W, focal, batch_rays, k):
    # 初始化两个空列表，用于存储每个光线批次的预期不确定性和后期不确定性
    pres = []
    posts = []
    
    # 计算光线的总数 N
    N = H * W
    # 计算每个批次中光线的数量 n
    n = batch_rays.shape[1] // N
    
    # 遍历每个光线批次
    for i in range(n):
        # 在不计算梯度的情况下进行渲染，以节省内存
        with torch.no_grad():
            # 渲染当前批次的光线，获取 RGB、深度、透明度、不确定性和额外信息
            rgb, weights, depth, uncert, raw = render_active(
                H, W, focal, chunk=args.chunk, 
                rays=batch_rays[:, i * N:i * N + N, :],  
                verbose=True, retraw=True,  
                **render_kwargs_train
            )

        # 计算渲染的不确定性，避免除以零
        uncert_render = uncert.reshape(-1, H * W, 1) + 1e-9
        # 获取每个光线的预测不确定性，并避免除以零
        uncert_pts = raw.reshape(-1, H * W, args.N_samples + args.N_importance) + 1e-9
        # 获取每个光线的权重
        weight_pts = weights.reshape(-1, H * W, args.N_samples + args.N_importance)

        # 计算先验不确定性的总和
        pre = uncert_pts.sum([1, 2])
        # 计算后验不确定性的总和 ps：先验概率和后验概率
        post = (1. / (1. / uncert_pts + weight_pts * weight_pts / uncert_render)).sum([1, 2])
        
        # 将先验和后验不确定性添加到列表中
        pres.append(pre)
        posts.append(post)
    
    # 将所有先验不确定性和后验不确定性合并为一个张量
    pres = torch.cat(pres, 0)
    posts = torch.cat(posts, 0)
    
    # 计算先验不确定性和后验不确定性之间的差值，并选择差值最大的 k 个光线的索引
    index = torch.topk(pres - posts, k)[1].cpu().numpy()

    # 返回选择的光线索引
    return index


def render_rays_active(
    ray_batch,
    model,
    featmaps,
    projector,
    N_samples,
    inv_uniform=False,
    N_importance=0,
    det=False,
    white_bkgd=False,
    ret_alpha=False,
    single_net=True,
):
    """
    渲染光线函数
    参数:
    ray_batch: 包含光线信息的字典 {'ray_o': [N_rays, 3] 光线原点, 'ray_d': [N_rays, 3] 光线方向, 'view_dir': [N_rays, 2] 视角方向}
    model: 包含网络模型的字典 {'net_coarse': 粗网络, 'net_fine': 细网络}
    featmaps: 特征图列表
    projector: 投影器对象
    N_samples: 每条光线上的采样点数(粗网络和细网络都使用)
    inv_uniform: 如果为True,则对粗网络均匀采样逆深度
    N_importance: 通过重要性采样产生的每条光线的额外采样点数(用于细网络)
    det: 如果为True,将确定性地采样深度
    ret_alpha: 如果为True,将返回从注意力图推断出的'密度'值
    single_net: 如果为True,将使用单个网络,可以同时处理粗采样点和细采样点
    返回: {'outputs_coarse': 粗网络输出, 'outputs_fine': 细网络输出}
    """

    # 初始化返回字典
    ret = {"outputs_coarse": None, "outputs_fine": None}
    # 获取光线原点和方向
    ray_o, ray_d = ray_batch["ray_o"], ray_batch["ray_d"]

    # 沿相机光线采样点
    # pts: [N_rays, N_samples, 3] - 采样点的3D坐标
    # z_vals: [N_rays, N_samples] - 采样点的深度值
    pts, z_vals = sample_along_camera_ray(
        ray_o=ray_o,
        ray_d=ray_d,
        depth_range=ray_batch["depth_range"],
        N_samples=N_samples,
        inv_uniform=inv_uniform,
        det=det,
    )

    # 获取光线数量和每条光线的采样点数
    N_rays, N_samples = pts.shape[:2]
    
    # 计算每个采样点的RGB特征、光线差异和掩码
    rgb_feat, ray_diff, mask = projector.compute(
        pts,
        ray_batch["camera"],
        ray_batch["src_rgbs"],
        ray_batch["src_cameras"],
        featmaps=featmaps[0],
    )  # [N_rays, N_samples, N_views, x]

    # 使用粗网络预测RGB值和不确定性
    rgb, uncert, raw = model.net_coarse(rgb_feat, ray_diff, mask, pts, ray_d)
    if ret_alpha:
        # 如果需要返回alpha值,将RGB和权重分开
        rgb, weights = rgb[:, 0:3], rgb[:, 3:]
        # 计算深度图
        depth_map = torch.sum(weights * z_vals, dim=-1)
    else:
        weights = None
        depth_map = None
    # 存储粗网络的输出
    ret["outputs_coarse"] = {"rgb": rgb, "weights": weights, "depth": depth_map, "uncert": uncert}

    # 如果需要细网络
    if N_importance > 0:
        # 分离权重以解耦粗网络和细网络
        weights = ret["outputs_coarse"]["weights"].clone().detach()  # [N_rays, N_samples]
        # 采样细网络的点
        pts, z_vals = sample_fine_pts(
            inv_uniform, N_importance, det, N_samples, ray_batch, weights, z_vals
        )

        # 计算细采样点的特征
        rgb_feat_sampled, ray_diff, mask = projector.compute(
            pts,
            ray_batch["camera"],
            ray_batch["src_rgbs"],
            ray_batch["src_cameras"],
            featmaps=featmaps[1],
        )

        # 使用相应的网络(单网络或细网络)预测RGB值和不确定性
        if single_net:
            rgb, uncert, raw = model.net_coarse(rgb_feat_sampled, ray_diff, mask, pts, ray_d)
        else:
            rgb, uncert, raw = model.net_fine(rgb_feat_sampled, ray_diff, mask, pts, ray_d)
        # 分离RGB和权重
        rgb, weights = rgb[:, 0:3], rgb[:, 3:]
        # 计算深度图
        depth_map = torch.sum(weights * z_vals, dim=-1)
        # 存储细网络的输出
        ret["outputs_fine"] = {"rgb": rgb, "weights": weights, "depth": depth_map, "uncert": uncert, "raw": raw}

    return ret


# def choose_new_k(H, W, focal, batch_rays, k, **render_kwargs_train):
    
#     pres = []
#     posts = []
#     N = H*W
#     n = batch_rays.shape[1] // N
#     for i in range(n):
#         with torch.no_grad():
#             rgb, disp, acc, uncert, alpha, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays[:,i*N:i*N+N,:],  verbose=True, retraw=True,  **render_kwargs_train)

#         uncert_render = uncert.reshape(-1, H*W, 1) + 1e-9
#         uncert_pts = extras['raw'][...,-1].reshape(-1, H*W, args.N_samples + args.N_importance) + 1e-9
#         weight_pts = extras['weights'].reshape(-1, H*W, args.N_samples + args.N_importance)

#         pre = uncert_pts.sum([1,2])
#         post = (1. / (1. / uncert_pts + weight_pts * weight_pts / uncert_render)).sum([1,2])
#         pres.append(pre)
#         posts.append(post)
    
#     pres = torch.cat(pres, 0)
#     posts = torch.cat(posts, 0)
#     index = torch.topk(pres-posts, k)[1].cpu().numpy()

#     return index

# def render_active(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
#                   near=0., far=1.,
#                   use_viewdirs=False, c2w_staticcam=None,
#                   **kwargs):
#     """渲染光线
#     参数:
#       H: int. 图像高度(像素)。
#       W: int. 图像宽度(像素)。
#       focal: float. 针孔相机的焦距。
#       chunk: int. 同时处理的最大光线数。用于控制最大内存使用量,不影响最终结果。
#       rays: 形状为[2, batch_size, 3]的数组。每个batch样本的光线原点和方向。
#       c2w: 形状为[3, 4]的数组。相机到世界的变换矩阵。
#       ndc: bool. 如果为True,则在NDC坐标系中表示光线原点和方向。
#       near: float或形状为[batch_size]的数组。光线的最近距离。
#       far: float或形状为[batch_size]的数组。光线的最远距离。
#       use_viewdirs: bool. 如果为True,在模型中使用空间中点的观察方向。
#       c2w_staticcam: 形状为[3, 4]的数组。如果不为None,则使用此变换矩阵作为相机,同时使用其他c2w参数作为观察方向。
#     返回:
#       rgb_map: [batch_size, 3]. 光线的预测RGB值。
#       disp_map: [batch_size]. 视差图。深度的倒数。
#       acc_map: [batch_size]. 沿光线累积的不透明度(alpha)。
#       extras: 包含render_rays()返回的所有内容的字典。
#     """
#     if c2w is not None:
#         # 特殊情况:渲染完整图像
#         rays_o, rays_d = get_rays(H, W, focal, c2w)
#     else:
#         # 使用提供的光线batch
#         rays_o, rays_d = rays

#     if use_viewdirs:
#         # 提供光线方向作为输入
#         viewdirs = rays_d
#         if c2w_staticcam is not None:
#             # 特殊情况:可视化viewdirs的效果
#             rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
#         viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
#         viewdirs = torch.reshape(viewdirs, [-1,3]).float()

#     sh = rays_d.shape # [..., 3]
#     if ndc:
#         # 对于前向场景
#         rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

#     # 创建光线batch
#     rays_o = torch.reshape(rays_o, [-1,3]).float()
#     rays_d = torch.reshape(rays_d, [-1,3]).float()

#     near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
#     rays = torch.cat([rays_o, rays_d, near, far], -1)
#     if use_viewdirs:
#         rays = torch.cat([rays, viewdirs], -1)

#     # 渲染并重塑
#     all_ret = batchify_rays(rays, chunk, **kwargs)
#     for k in all_ret:
#         k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
#         all_ret[k] = torch.reshape(all_ret[k], k_sh)

#     k_extract = ['rgb', 'weights', 'depth', 'uncert', 'raw ']
#     ret_list = [all_ret[k] for k in k_extract]
#     ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
#     return ret_list + [ret_dict]


# def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
#     """Render rays in smaller minibatches to avoid OOM.
#     """
#     all_ret = {}
#     for i in range(0, rays_flat.shape[0], chunk):
#         featmaps = model.feature_net(rays_flat[i:i+chunk].squeeze(0).permute(0, 3, 1, 2))
#         ret = render_rays(rays_flat[i:i+chunk], featmaps, **kwargs)
#         for k in ret:
#             if k not in all_ret:
#                 all_ret[k] = []
#             all_ret[k].append(ret[k])

#     all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
#     return all_ret

# def ndc_rays(H, W, focal, near, rays_o, rays_d):
#     # 将射线转换到标准化设备坐标(NDC)空间
#     # 将射线原点移动到近平面
#     t = -(near + rays_o[...,2]) / rays_d[...,2]
#     rays_o = rays_o + t[...,None] * rays_d
    
#     # 投影变换
#     o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
#     o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
#     o2 = 1. + 2. * near / rays_o[...,2]

#     d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
#     d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
#     d2 = -2. * near / rays_o[...,2]
    
#     rays_o = torch.stack([o0,o1,o2], -1)
#     rays_d = torch.stack([d0,d1,d2], -1)
    
#     return rays_o, rays_d

def render_rays(
    ray_batch,
    model,
    featmaps,
    projector,
    N_samples,
    inv_uniform=False,
    N_importance=0,
    det=False,
    white_bkgd=False,
    ret_alpha=False,
    single_net=True,
):
    """
    渲染光线函数
    参数:
    ray_batch: 包含光线信息的字典 {'ray_o': [N_rays, 3] 光线原点, 'ray_d': [N_rays, 3] 光线方向, 'view_dir': [N_rays, 2] 视角方向}
    model: 包含网络模型的字典 {'net_coarse': 粗网络, 'net_fine': 细网络}
    featmaps: 特征图列表
    projector: 投影器对象
    N_samples: 每条光线上的采样点数(粗网络和细网络都使用)
    inv_uniform: 如果为True,则对粗网络均匀采样逆深度
    N_importance: 通过重要性采样产生的每条光线的额外采样点数(用于细网络)
    det: 如果为True,将确定性地采样深度
    ret_alpha: 如果为True,将返回从注意力图推断出的'密度'值
    single_net: 如果为True,将使用单个网络,可以同时处理粗采样点和细采样点
    返回: {'outputs_coarse': 粗网络输出, 'outputs_fine': 细网络输出}
    """

    # 初始化返回字典
    ret = {"outputs_coarse": None, "outputs_fine": None}
    # 获取光线原点和方向
    ray_o, ray_d = ray_batch["ray_o"], ray_batch["ray_d"]

    # 沿相机光线采样点
    # pts: [N_rays, N_samples, 3] - 采样点的3D坐标
    # z_vals: [N_rays, N_samples] - 采样点的深度值
    pts, z_vals = sample_along_camera_ray(
        ray_o=ray_o,
        ray_d=ray_d,
        depth_range=ray_batch["depth_range"],
        N_samples=N_samples,
        inv_uniform=inv_uniform,
        det=det,
    )

    # 获取光线数量和每条光线的采样点数
    N_rays, N_samples = pts.shape[:2]
    
    # 计算每个采样点的RGB特征、光线差异和掩码
    rgb_feat, ray_diff, mask = projector.compute(
        pts,
        ray_batch["camera"],
        ray_batch["src_rgbs"],
        ray_batch["src_cameras"],
        featmaps=featmaps[0],
    )  # [N_rays, N_samples, N_views, x]

    # 使用粗网络预测RGB值和不确定性
    rgb, uncert, raw = model.net_coarse(rgb_feat, ray_diff, mask, pts, ray_d)
    if ret_alpha:
        # 如果需要返回alpha值,将RGB和权重分开
        rgb, weights = rgb[:, 0:3], rgb[:, 3:]
        # 计算深度图
        depth_map = torch.sum(weights * z_vals, dim=-1)
    else:
        weights = None
        depth_map = None
    # 存储粗网络的输出
    ret["outputs_coarse"] = {"rgb": rgb, "weights": weights, "depth": depth_map, "uncert": uncert, "raw": raw}

    # 如果需要细网络
    #print('N_importance:',N_importance)
    if N_importance > 0:
        # 分离权重以解耦粗网络和细网络
        weights = ret["outputs_coarse"]["weights"].clone().detach()  # [N_rays, N_samples]
        # 采样细网络的点
        pts, z_vals = sample_fine_pts(
            inv_uniform, N_importance, det, N_samples, ray_batch, weights, z_vals
        )

        # 计算细采样点的特征
        rgb_feat_sampled, ray_diff, mask = projector.compute(
            pts,
            ray_batch["camera"],
            ray_batch["src_rgbs"],
            ray_batch["src_cameras"],
            featmaps=featmaps[1],
        )

        # 使用相应的网络(单网络或细网络)预测RGB值和不确定性
        if single_net:
            rgb, uncert, raw = model.net_coarse(rgb_feat_sampled, ray_diff, mask, pts, ray_d)
        else:
            rgb, uncert, raw = model.net_fine(rgb_feat_sampled, ray_diff, mask, pts, ray_d)
        # 分离RGB和权重
        rgb, weights = rgb[:, 0:3], rgb[:, 3:]
        # 计算深度图
        depth_map = torch.sum(weights * z_vals, dim=-1)
        # 存储细网络的输出
        ret["outputs_fine"] = {"rgb": rgb, "weights": weights, "depth": depth_map, "uncert": uncert, "raw": raw}

    return ret
