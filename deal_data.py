import json
import os

def split_transforms_json(input_path):
    # 读取原始JSON文件
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # 创建两个新的数据结构
    train_data = {
        "camera_angle_x": data["camera_angle_x"],
        "frames": data["frames"][:80]  # 前80帧
    }
    
    holdout_data = {
        "camera_angle_x": data["camera_angle_x"],
        "frames": data["frames"][80:404]  # 后304帧
    }
    
    # 获取输出路径
    dir_path = os.path.dirname(input_path)
    holdout_path = os.path.join(dir_path, "transforms_holdout.json")
    
    # 保存文件
    with open(input_path, 'w') as f:
        json.dump(train_data, f, indent=4)
        
    with open(holdout_path, 'w') as f:
        json.dump(holdout_data, f, indent=4)
        
    print(f"已将数据分割为两个文件：")
    print(f"训练集（80帧）：{input_path}")
    print(f"验证集（324帧）：{holdout_path}")

# 处理nerf_synthetic数据集中的所有场景
scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship' ]
base_dir = "./nerf_synthetic"  # 根据实际路径调整

for scene in scenes:
    input_file = os.path.join(base_dir, scene, "transforms_train.json")
    if os.path.exists(input_file):
        print(f"\n处理场景：{scene}")
        split_transforms_json(input_file)
    else:
        print(f"警告：找不到场景 {scene} 的文件")


import os
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from gnt.data_loaders import dataset_dict
from gnt.render_image import render_single_image
# from gnt.model import GNTModel
# from gnt.sample_ray import RaySamplerSingleImage
# from utils import img_HWC2CHW, colorize, img2psnr, lpips, ssim
import config
# import torch.distributed as dist
# from gnt.projection import Projector
# from gnt.data_loaders.create_training_dataset import create_training_dataset
# import imageio


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


@torch.no_grad()
def renders(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create holdout dataset
    dataset = dataset_dict[args.eval_dataset](args, "holdout", scenes=args.eval_scenes)
    loader = DataLoader(dataset, batch_size=1)
    iterator = iter(loader)

    # Create GNT model
    model = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    # create projector
    projector = Projector(device=device)

    indx = 0
    while True:
        try:
            data = next(iterator)
        except:
            break
        if args.local_rank == 0:
            tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
            # psnr_curr_img, lpips_curr_img, ssim_curr_img = log_view(
            ret = log_view(
                indx,
                args,
                model,
                tmp_ray_sampler,
                projector,
                gt_img,
                # render_stride=args.render_stride,
                # prefix="val/",
                # out_folder=out_folder,
                # ret_alpha=args.N_importance > 0,
                # single_net=args.single_net,
            )
            # psnr_scores.append(psnr_curr_img)
            # lpips_scores.append(lpips_curr_img)
            # ssim_scores.append(ssim_curr_img)
            # torch.cuda.empty_cache()
            # indx += 1
    # print("Average PSNR: ", np.mean(psnr_scores))
    # print("Average LPIPS: ", np.mean(lpips_scores))
    # print("Average SSIM: ", np.mean(ssim_scores))


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
        print("============= ret: ==============")
        print(ret)
    return ret

    # average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    # if args.render_stride != 1:
    #     gt_img = gt_img[::render_stride, ::render_stride]
    #     average_im = average_im[::render_stride, ::render_stride]

    # rgb_gt = img_HWC2CHW(gt_img)
    # average_im = img_HWC2CHW(average_im)

    # rgb_coarse = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())
    # if "depth" in ret["outputs_coarse"].keys():
    #     depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()
    #     depth_coarse = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    # else:
    #     depth_coarse = None

    # if ret["outputs_fine"] is not None:
    #     rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
    #     if "depth" in ret["outputs_fine"].keys():
    #         depth_pred = ret["outputs_fine"]["depth"].detach().cpu()
    #         depth_fine = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    # else:
    #     rgb_fine = None
    #     depth_fine = None

    # rgb_coarse = rgb_coarse.permute(1, 2, 0).detach().cpu().numpy()
    # filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}_coarse.png".format(global_step))
    # imageio.imwrite(filename, rgb_coarse)

    # if depth_coarse is not None:
    #     depth_coarse = depth_coarse.permute(1, 2, 0).detach().cpu().numpy()
    #     filename = os.path.join(
    #         out_folder, prefix[:-1] + "_{:03d}_coarse_depth.png".format(global_step)
    #     )
    #     imageio.imwrite(filename, depth_coarse)

    # if rgb_fine is not None:
    #     rgb_fine = rgb_fine.permute(1, 2, 0).detach().cpu().numpy()
    #     filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}_fine.png".format(global_step))
    #     imageio.imwrite(filename, rgb_fine)

    # if depth_fine is not None:
    #     depth_fine = depth_fine.permute(1, 2, 0).detach().cpu().numpy()
    #     filename = os.path.join(
    #         out_folder, prefix[:-1] + "_{:03d}_fine_depth.png".format(global_step)
    #     )
    #     imageio.imwrite(filename, depth_fine)

    # # write scalar
    # pred_rgb = (
    #     ret["outputs_fine"]["rgb"]
    #     if ret["outputs_fine"] is not None
    #     else ret["outputs_coarse"]["rgb"]
    # )
    # pred_rgb = torch.clip(pred_rgb, 0.0, 1.0)
    # lpips_curr_img = lpips(pred_rgb, gt_img, format="HWC").item()
    # ssim_curr_img = ssim(pred_rgb, gt_img, format="HWC").item()
    # psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    # print(prefix + "psnr_image: ", psnr_curr_img)
    # print(prefix + "lpips_image: ", lpips_curr_img)
    # print(prefix + "ssim_image: ", ssim_curr_img)
    # return psnr_curr_img, lpips_curr_img, ssim_curr_img


if __name__ == "__main__":
    # parser = config.config_parser()
    # parser.add_argument("--run_val", action="store_true", help="run on val set")
    # args = parser.parse_args()

    # if args.distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl", init_method="env://")
    #     synchronize()

    # renders(args)

    # holdout_datest = dataset_dict['nerf_synthetic_holdout'](args, scenes='chair')
    # hold_loader = DataLoader(holdout_datest, batch_size=1)
    # print("#"*78)
    pst = np.array([ 3, 4, 5, 6,6 ,6 ,7, 9,7])

    print(pst[[6,3,2,1]].tolist())

    # print(holdout_datest.render_intrinsics)
    # print(holdout_datest.render_intrinsics)
    # print(type(holdout_datest.render_intrinsics))
    # print(len(holdout_datest.render_intrinsics))
    # print(type(holdout_datest.render_poses[0]))
    # print(holdout_datest.render_poses[0])
    # print(holdout_datest.render_rgb_files)
    # print(len(holdout_datest))
    # print(len(hold_loader))
    # print([i for i in range(80)])
    # idesst = [71, 29, 68, 23, 7, 1]
    # render_rgb_files, render_poses, render_intrinsics = holdout_datest.delete_indexs(idesst)

    # print(render_rgb_files)
    # print(render_poses)
    # print(render_intrinsics)

    # hold_loader = DataLoader(holdout_datest, batch_size=1)
    # print("#"*78)
    # print(len(holdout_datest))

    # for i, data in enumerate(hold_loader):
    #     # print(i)
    #     # print(type(data))
    #     print(data.keys())
    #     for k in data.keys():
    #         if isinstance(data[k], torch.Tensor) and k != 'ids':
    #             print(k, data[k].shape)
    #         elif k == 'ids':
    #             print(k, data[k].tolist())
    #         else:
    #             print(k, data[k])
    #     print("#"*78)
