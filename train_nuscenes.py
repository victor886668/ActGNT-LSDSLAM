import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from gnt.data_loaders import dataset_dict
from gnt.render_ray import render_rays, choose_new_k
from gnt.render_image import render_single_image
from gnt.model import GNTModel
from gnt.sample_ray import RaySamplerSingleImage
from gnt.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr
import config
import torch.distributed as dist
from gnt.projection import Projector
from gnt.data_loaders.create_training_dataset import create_training_dataset, create_holdout_dataset
import imageio


def get_rays_np(H, W, focal, c2w):
    # 使用numpy生成相机射线的函数(与get_rays功能相同)
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def choose_new_ks(holdout_loader, args, model, projector):
    # torch.cuda.empty_cache()
    # model.switch_to_eval()

    model.switch_to_eval()
    device = "cuda:{}".format(args.local_rank)
    pres = []
    posts = []
    all_ids = []
    
    for hold_data in holdout_loader:
        # print("#"*78)
        # print(hold_data['ids'])
        all_ids.extend(hold_data.pop('ids'))
        with torch.no_grad():
            # ray_batch = ray_sampler.get_all()
            # if model.feature_net is not None:
            #     featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            # else:
            #     featmaps = [None, None]
            # print("到这来了吗？")
            # ret = render_single_image(
            #     ray_sampler=ray_sampler,
            #     ray_batch=ray_batch,
            #     model=model,
            #     projector=projector,
            #     chunk_size=args.chunk_size,
            #     N_samples=args.N_samples,
            #     inv_uniform=args.inv_uniform,
            #     det=True,
            #     N_importance=args.N_importance,
            #     white_bkgd=args.white_bkgd,
            #     render_stride=args.render_stride,
            #     featmaps=featmaps,
            #     ret_alpha=args.N_importance > 0,
            #     single_net=args.single_net
            # )

            ray_sampler = RaySamplerSingleImage(hold_data, device)
            N_rand = int(
                1.0 * args.N_rand * args.num_source_views / hold_data["src_rgbs"][0].shape[0]
            )

            ray_batch = ray_sampler.random_sample(
                N_rand,
                sample_mode=args.sample_mode,
                center_ratio=args.center_ratio,
            )

            #print("ray_batch: ", ray_batch['src_rgbs'].shape)
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

        # print("#"*78)
        # print(ret.keys())
        # print(ret['outputs_fine'].keys())
        ret_tp = ret['outputs_fine']
        weights, uncert, raw = ret_tp['weights'], ret_tp['uncert'], ret_tp['raw']
        # print("weights: ", weights.shape)
        # print("uncert: ", uncert.shape)
        # print("raw : ", raw.shape)

        # 计算渲染的不确定性，避免除以零
        uncert_render = uncert.reshape(-1) + 1e-9
        # 获取每个光线的预测不确定性，并避免除以零
        # uncert_pts = raw.reshape(-1, 512, args.N_samples + args.N_importance) + 1e-9
        uncert_pts = raw.reshape(-1, 512) + 1e-9

        # 获取每个光线的权重
        # weight_pts = weights[...,-1].reshape(-1, 512, args.N_samples + args.N_importance)
        weight_pts = weights[...,-1].reshape(-1, 512)

        # print("weight_pts: ", weight_pts.shape)
        # print("uncert_pts: ", uncert_pts.shape)
        # print("uncert_render : ", uncert_render.shape)

        # 计算先验不确定性的总和
        pre = uncert_pts.sum([0, 1])
        # print("pre: ", pre)
        # print("#"*67)
        # 计算后验不确定性的总和 ps：先验概率和后验概率
        post = (1. / (1. / uncert_pts + weight_pts * weight_pts / uncert_render)).sum([0, 1])
        # print("post : ", post)
        # 将先验和后验不确定性添加到列表中
        pres.append(pre.cpu().data)
        posts.append(post.cpu().data)
        # torch.cuda.empty_cache()
    
    # print("pres shape: ", pres)
    # print("posts shape: ", posts)
    # 将所有先验不确定性和后验不确定性合并为一个张量
    pres = torch.Tensor(pres)
    posts = torch.Tensor(posts)

    # print("pres shape: ", pres.shape)
    # print("posts shape: ", posts.shape)

    # print("#"*78)
    # print("pres shape: ", pres)
    # print("posts shape: ", posts)
    
    
    # 计算先验不确定性和后验不确定性之间的差值，并选择差值最大的 k 个光线的索引
    index = torch.topk(pres - posts, args.choose_k)[1].cpu().tolist()
    # 返回选择的光线索引
    return np.array(all_ids)[index].tolist()

    

# def holdout_deal(args):
#     device = "cuda:{}".format(args.local_rank)
#     out_folder = os.path.join(args.rootdir, "out", args.expname)
#     print("outputs will be saved to {}".format(out_folder))
#     os.makedirs(out_folder, exist_ok=True)
#     holdout_datest = dataset_dict['nerf_synthetic_holdout'](args, scenes=args.eval_scenes)
#     hold_loader = DataLoader(holdout_datest, batch_size=1)

#     # Create GNT model
#     model = GNTModel(
#         args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
#     )
#     # create projector
#     projector = Projector(device=device)

#     choose_new_ks(hold_loader, args, model, projector, device)

def write_log(out_folder, m='w', out='========================== 开始训练 ======================\n'):
    f = os.path.join(out_folder, "logging.txt")
    if m == 'w':
        with open(f, m) as file:
            file.write("{}\n".format(args.config))
            file.write('========================== 开始训练 ======================\n')
            file.close()
    else:
        with open(f, 'a') as file:
            file.write(f"{out}")
            file.close()
def train(args):
    print('args.rootdir:',args.rootdir)
    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out_neft", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    # 创建写日志
    write_log(out_folder, m='w')

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    holdout_dataset, holdout_sampler = create_training_dataset(args,model='holdout')
    print('test:',len(holdout_dataset))
    # holdout_dataset, holdout_sampler = create_holdout_dataset(args)
    # all_pose = train_dataset.render_poses + holdout_dataset.render_poses

    # holdout_datest = dataset_dict['nerf_synthetic_holdout'](args, scenes=args.eval_scenes)
    # hold_loader = DataLoader(holdout_datest, batch_size=1)

    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        worker_init_fn=lambda _: np.random.seed(),
        num_workers=1,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
    )
    
    hold_loader = torch.utils.data.DataLoader(
        holdout_dataset,
        batch_size=1,
        worker_init_fn=lambda _: np.random.seed(),
        num_workers=1,
        pin_memory=True,
        sampler=holdout_sampler,
        shuffle=True if holdout_sampler is None else False,
    )
    
    

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)
    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # create holdout dataset
    # holdout_loader = DataLoader(holdout_dataset, batch_size=1)
    
    # Create GNT model
    model = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    # create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 1
    all_train_num = len(holdout_dataset)+len(train_dataset)
    print('all_train_num:',all_train_num)
    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()

        # 定义执行active的部分
        
        if args.local_rank == 0:
            # 开始active部分
            # if epoch & args.active_iter:
            print(f" {epoch} % {args.active_iter} 等于 多少：", epoch % args.active_iter)
            if epoch % args.active_iter == 0:
                print("Start Active 部分")

                if len(holdout_dataset) == 0:
                    print("No holdout dataset!")
                elif len(holdout_dataset) <= (args.choose_k+1):#少于9个
                    if len(train_dataset) >= all_train_num:
                        print("No holdout dataset add !")
                        # continue
                    else:
                        render_rgb_files, render_poses, render_intrinsics = holdout_dataset.render_rgb_files, holdout_dataset.render_poses, holdout_dataset.render_intrinsics

                        # 更新train dataset 数据
                        train_dataset.add_mixdata(render_rgb_files, render_poses, render_intrinsics)

                        train_sampler = (
                            torch.utils.data.distributed.DistributedSampler(train_dataset)
                            if args.distributed
                            else None
                        )

                        train_loader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=1,
                            worker_init_fn=lambda _: np.random.seed(),
                            num_workers=1,
                            pin_memory=True,
                            sampler=train_sampler,
                            shuffle=True if train_sampler is None else False,
                        )

                else:
                    
                    #print('hold_loader:',len(hold_loader))
                    indexs = choose_new_ks(hold_loader, args, model, projector)
                    
                    # 在hold_dataset 删除掉已经选出来的元素
                    render_rgb_files, render_poses, render_intrinsics = holdout_dataset.delete_indexs(indexs)
                    #hold_loader = DataLoader(holdout_dataset, batch_size=1)

                    # 更新train dataset 数据
                    train_dataset.add_mixdata(render_rgb_files, render_poses, render_intrinsics)

                    train_sampler = (
                        torch.utils.data.distributed.DistributedSampler(train_dataset)
                        if args.distributed
                        else None
                    )

                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=1,
                        worker_init_fn=lambda _: np.random.seed(),
                        num_workers=1,
                        pin_memory=True,
                        sampler=train_sampler,
                        shuffle=True if train_sampler is None else False,
                    )
                    
                    hold_loader = torch.utils.data.DataLoader(
                        holdout_dataset,
                        batch_size=1,
                        worker_init_fn=lambda _: np.random.seed(),
                        num_workers=1,
                        pin_memory=True,
                        sampler=holdout_sampler,
                        shuffle=True if holdout_sampler is None else False,
                    )
                    
                    
                    

                    # 切换回训练模式
                    model.switch_to_train()

                    # f = os.path.join(args.rootdir, "out", args.expname, 'args.txt')
                    logstr = f"目前删除了{str(indexs)}个holdout数据，目前剩余{len(holdout_dataset)}个holdout数据\n目前增加了{len(indexs)}个train数据，目前总共{len(train_dataset)}个train数据\n"
                    write_log(out_folder, m='a', out=logstr)

                    # with open(f, 'a') as file:
                    #     file.write(f"目前删除了{str(indexs)}个holdout数据，目前剩余{len(holdout_datest)}个holdout数据\n目前增加了{len(indexs)}个train数据，目前总共{len(train_dataset)}个train数据\n")
                    #     file.write(f"目前增加了{len(indexs)}个train数据，目前总共{len(train_dataset)}个train数据\n")
        
        n = 0
        print('len:',len(train_loader))
        for train_data in train_loader:
            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop

            # load training rays
            #print('shape:',train_data["src_rgbs"][0].shape)
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
            # print('ret:',ret.keys())
            # print('ret:',ret['outputs_coarse'].keys())
            

            # compute loss
            model.optimizer.zero_grad()
            loss, scalars_to_log = criterion(ret["outputs_coarse"], ray_batch, scalars_to_log)

            if ret["outputs_fine"] is not None:
                fine_loss, scalars_to_log = criterion(
                    ret["outputs_fine"], ray_batch, scalars_to_log
                )
                loss += fine_loss

            loss.backward()
            scalars_to_log["loss"] = loss.item()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log["lr"] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0


            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret["outputs_coarse"]["rgb"], ray_batch["rgb"]).item()
                    scalars_to_log["train/coarse-loss"] = mse_error
                    scalars_to_log["train/coarse-psnr-training-batch"] = mse2psnr(mse_error)

                    if ret["outputs_fine"] is not None:
                        mse_error = img2mse(ret["outputs_fine"]["rgb"], ray_batch["rgb"]).item()
                        scalars_to_log["train/fine-loss"] = mse_error
                        scalars_to_log["train/fine-psnr-training-batch"] = mse2psnr(mse_error)

                    logstr = "{} Epoch: {}  step: {} \n".format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += " {}: {:.6f}".format(k, scalars_to_log[k])
                    print(logstr)
                    # 时间纪录
                    logstr += "each iter time {:.05f} seconds\n".format(dt)
                    print("each iter time {:.05f} seconds".format(dt))
                    write_log(out_folder, m='a', out=logstr)

                if global_step % args.i_weights == 0:
                    print("Saving checkpoints at {} to {}...".format(global_step, out_folder))
                    fpath = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))
                    model.save_model(fpath)
                    write_log(out_folder, m='a', out=f"Saved checkpoints at model_{global_step:06d}.pth  to {out_folder}\n")

                if global_step % args.i_img == 0:
                    print("Logging a random validation view...")
                    val_data = next(val_loader_iterator)
                    tmp_ray_sampler = RaySamplerSingleImage(
                        val_data, device, render_stride=args.render_stride
                    )
                    H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                    gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                    log_view(
                        global_step,
                        args,
                        model,
                        tmp_ray_sampler,
                        projector,
                        gt_img,
                        render_stride=args.render_stride,
                        prefix="val/",
                        out_folder=out_folder,
                        ret_alpha=args.N_importance > 0,
                        single_net=args.single_net,
                    )
                    torch.cuda.empty_cache()

                    # print("Logging current training view...")
                    # tmp_ray_train_sampler = RaySamplerSingleImage(
                    #     train_data, device, render_stride=1
                    # )
                    # H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                    # gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                    # log_view(
                    #     global_step,
                    #     args,
                    #     model,
                    #     tmp_ray_train_sampler,
                    #     projector,
                    #     gt_img,
                    #     render_stride=1,
                    #     prefix="train/",
                    #     out_folder=out_folder,
                    #     ret_alpha=args.N_importance > 0,
                    #     single_net=args.single_net,
                    # )
            global_step += 1
            #print('global_step:',global_step)
            # if global_step > model.start_step + args.n_iters + 1:
            #     break
        epoch += 1
        print('epoch:',epoch)
    # 保存最后一次模型权重
    
    print("Saving checkpoints at last time...")
    fpath = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))
    model.save_model(fpath)
    write_log(out_folder, m='a', out=f"Saved checkpoints at model_{global_step:06d}.pth  to {out_folder}\n")


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    render_stride=1,
    prefix="",
    out_folder="",
    ret_alpha=False,
    single_net=True,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]
        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps,
            ret_alpha=ret_alpha,
            single_net=single_net,
        )
    #print('keys:',ret["outputs_coarse"].keys())

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3 * w_max)
    rgb_im[:, : average_im.shape[-2], : average_im.shape[-1]] = average_im
    rgb_im[:, : rgb_gt.shape[-2], w_max : w_max + rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, : rgb_pred.shape[-2], 2 * w_max : 2 * w_max + rgb_pred.shape[-1]] = rgb_pred
    
    if "depth" in ret["outputs_coarse"].keys():
        print('111')
        depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()
        depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    else:
        depth_im = None

    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, : rgb_fine.shape[-2], : rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_pred = torch.cat((depth_pred, ret["outputs_fine"]["depth"].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))

    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy()
    # 转成rgb格式
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    rgb_im = to8b(rgb_im)
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}.png".format(global_step))
    imageio.imwrite(filename, rgb_im)

    if depth_im is not None:
        
        depth_im = depth_im.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(out_folder, prefix[:-1] + "depth_{:03d}.png".format(global_step))
        imageio.imwrite(filename, to8b(depth_im))

    # write scalar
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None
        else ret["outputs_coarse"]["rgb"]
    )
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    print("psnr_curr_img: ", psnr_curr_img)
    print(prefix + "psnr_image: ", psnr_curr_img)
    write_log(out_folder, m='a', out=f" val psnr: {psnr_curr_img:.4f}\n")
    model.switch_to_train()


if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    # if args.distributed:
    #     torch.distributed.init_process_group(backend="nccl", init_method="env://")
    #     args.local_rank = int(os.environ.get("LOCAL_RANK"))
    #     torch.cuda.set_device(args.local_rank)

    train(args)
    # holdout_deal(args)
