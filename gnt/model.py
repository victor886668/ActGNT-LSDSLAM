# 导入必要的库
import torch
import os
from gnt.transformer_network import GNT  # 导入GNT网络
from gnt.feature_network import ResUNet  # 导入ResUNet特征提取网络


# 移除模型的并行包装器
def de_parallel(model):
    return model.module if hasattr(model, "module") else model


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class GNTModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        # 初始化GNT模型
        self.args = args  # 保存参数
        device = torch.device("cuda:{}".format(args.local_rank))  # 设置设备
        # 创建粗糙GNT网络
        self.net_coarse = GNT(
            args,
            in_feat_ch=self.args.coarse_feat_dim,  # 粗糙特征维度
            posenc_dim=3 + 3 * 2 * 10,  # 位置编码维度
            viewenc_dim=3 + 3 * 2 * 10,  # 视角编码维度
            ret_alpha=args.N_importance > 0,  # 是否返回alpha值
        ).to(device)
        # 是否使用单一网络 - 训练单一网络可用于粗糙和精细采样
        if args.single_net:
            self.net_fine = None  # 不创建精细网络
        else:
            # 创建精细GNT网络
            self.net_fine = GNT(
                args,
                in_feat_ch=self.args.fine_feat_dim,  # 精细特征维度
                posenc_dim=3 + 3 * 2 * 10,  # 位置编码维度
                viewenc_dim=3 + 3 * 2 * 10,  # 视角编码维度
                ret_alpha=True,  # 返回alpha值
            ).to(device)

        # 创建特征提取网络
        self.feature_net = ResUNet(
            coarse_out_ch=self.args.coarse_feat_dim,  # 粗糙输出通道数
            fine_out_ch=self.args.fine_feat_dim,  # 精细输出通道数
            single_net=self.args.single_net,  # 是否使用单一网络
        ).to(device)

        # 优化器和学习率调度器
        learnable_params = list(self.net_coarse.parameters())  # 收集可学习参数
        learnable_params += list(self.feature_net.parameters())  # 添加特征网络参数
        if self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())  # 添加精细网络参数

        # 创建优化器
        if self.net_fine is not None:
            # 使用不同的学习率创建优化器
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},  # 粗糙网络参数
                    {"params": self.net_fine.parameters()},  # 精细网络参数
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},  # 特征网络参数
                ],
                lr=args.lrate_gnt,  # GNT网络的学习率
            )
        else:
            # 只优化粗糙网络和特征网络
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},  # 粗糙网络参数
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},  # 特征网络参数
                ],
                lr=args.lrate_gnt,  # GNT网络的学习率
            )

        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=args.lrate_decay_steps,  # 学习率衰减步长
            gamma=args.lrate_decay_factor  # 学习率衰减因子
        )

        # 加载检查点
        out_folder = os.path.join(args.rootdir, "out", args.expname)  # 输出文件夹路径
        self.start_step = self.load_from_ckpt(
            out_folder, 
            load_opt=load_opt,  # 是否加载优化器状态
            load_scheduler=load_scheduler  # 是否加载调度器状态
        )

        # 如果使用分布式训练
        if args.distributed:
            # 将网络转换为分布式数据并行模式
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse, 
                device_ids=[args.local_rank],  # 设备ID
                output_device=args.local_rank  # 输出设备
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net, 
                device_ids=[args.local_rank],  # 设备ID
                output_device=args.local_rank  # 输出设备
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine, 
                    device_ids=[args.local_rank],  # 设备ID
                    output_device=args.local_rank  # 输出设备
                )

    def switch_to_eval(self):
        # 切换到评估模式
        self.net_coarse.eval()  # 粗糙网络评估模式
        self.feature_net.eval()  # 特征网络评估模式
        if self.net_fine is not None:
            self.net_fine.eval()  # 精细网络评估模式

    def switch_to_train(self):
        # 切换到训练模式
        self.net_coarse.train()  # 粗糙网络训练模式
        self.feature_net.train()  # 特征网络训练模式
        if self.net_fine is not None:
            self.net_fine.train()  # 精细网络训练模式

    def save_model(self, filename):
        # 保存模型状态
        to_save = {
            "optimizer": self.optimizer.state_dict(),  # 保存优化器状态
            "scheduler": self.scheduler.state_dict(),  # 保存调度器状态
            "net_coarse": de_parallel(self.net_coarse).state_dict(),  # 保存粗糙网络状态
            "feature_net": de_parallel(self.feature_net).state_dict(),  # 保存特征网络状态
        }

        if self.net_fine is not None:
            to_save["net_fine"] = de_parallel(self.net_fine).state_dict()  # 保存精细网络状态

        torch.save(to_save, filename)  # 保存到文件

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        # 加载模型状态
        if self.args.distributed:
            # 分布式加载
            to_load = torch.load(filename, map_location="cuda:{}".format(self.args.local_rank))
        else:
            # 普通加载
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])  # 加载优化器状态
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])  # 加载调度器状态

        self.net_coarse.load_state_dict(to_load["net_coarse"])  # 加载粗糙网络状态
        self.feature_net.load_state_dict(to_load["feature_net"])  # 加载特征网络状态

        if self.net_fine is not None and "net_fine" in to_load.keys():
            self.net_fine.load_state_dict(to_load["net_fine"])  # 加载精细网络状态

    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        从现有检查点加载模型并返回当前步骤
        :param out_folder: 存储检查点的目录
        :return: 当前起始步骤
        """

        # 获取所有现有检查点
        ckpts = []
        print('out_folder:',out_folder)
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        # 如果指定了检查点路径且不强制使用最新检查点
        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # 加载指定的检查点
                ckpts = [self.args.ckpt_path]

        # 如果存在检查点且不禁止重新加载
        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]  # 获取最新检查点
            self.load_model(fpath, load_opt, load_scheduler)  # 加载模型
            step = int(fpath[-10:-4])  # 获取步骤数
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")  # 没有找到检查点，从头开始训练
            step = 0

        return step  # 返回当前步骤
