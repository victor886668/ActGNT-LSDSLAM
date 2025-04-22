# ActGNT-LSDSLAM
整体步骤如下
#step1 重命名文件:
python process_nucenes.py

#step2 colmap处理数据
# 直接看三章节，参考：https://zhuanlan.zhihu.com/p/657394640
# 这里有菜单选项，按顺序来
（0）先新建一个文件夹：colmap_process
（1）双击打开COLMAP.bat
（2）进入File》New Project，会有一个菜单
 点击new 在colmap_process文件夹下手动输入文件名1.db后，点击保存
  然后 Select，选择step1处理后的images_rename文件夹，点击Save。
（3）进入Processing》Feature extraction 只配置第一个选项 SIMPLE_PINHOLE，然后Extract 运行结束后关闭小窗口
（4）进入Processing》Feature matching》run 无需配置选项直接运行 运行结束后关闭小窗口
（5）进入Reconstruction》Start Reconstruction
（6）运行结束后进入 File》Export model 存入colmap_process\sparse\0，这里存有pose

#step3 colmap转llff所需pose:   依赖：colmap_read_model.py
# ./colmap_process：是刚处理colmap的路径
python imgs2poses.py ./colmap_process

#上述处理后会得到./colmap_process/poses_bounds.npy文件


#step4 整理成llff格式数据集
新建一个文件夹名例如为：llff_nuscenes
拷贝images_rename 至 llff_nuscenes 文件里，并将images_rename命名为images
拷贝colmap_process/sparse 至 llff_nuscenes 文件里
拷贝/colmap_process/poses_bounds.npy 至 llff_nuscenes 文件里


#step5 训练 activate+GNT
#已经把数据放在里面了，./data/nerf_llff_data/nuscenes

python train_nuscenes.py --config configs/gnt_llff_nuscenes.txt --train_scenes nuscenes --eval_scenes nuscenes

#step6 评估保存所有数据的不确定性图
python eval_nusecenes.py \
--config configs/gnt_llff_nuscenes.txt \
--eval_scenes nuscenes \
--expname gnt_llff_nuscenes \
--chunk_size 500 \
--run_val \
--N_samples 192 

#在out/gnt_llff_nuscenes/eval里保存了所有文件


#step7 整理lsd-slam所需要的数据
在step4 构建好的数据里增加一个calib.cfg的文件，这里已经提供data/7_llff_test/calib.cfg


#step8 分组实验
#exp1 基于不确定图选关键帧,编译运行lsd_slam_ws01
编译
cd /home/slam/lsd_slam_ws01
catkin_make -j6

编译成功后再运行

运行：
step1:
roscore

step2:
cd /home/slam/lsd_slam_ws01
source devel/setup.bash 
rosrun lsd_slam_viewer viewer

step3:
cd /home/slam/lsd_slam_ws01
source ./devel/setup.bash
rosrun lsd_slam_core dataset using _files:=/home/slam/data/nusceness_llff/images  _/calib:=/home/slam/data/7_llff_test/calib.cfg _hz:=2

#exp2 原始lsd关键帧选取:
#编译运行lsd_slam_ws02


#step9 评估结果
#用evo库进行
pip install evo --upgrade --no-binary evo

#在step8保存pose的路径执行命令
# ape
#lsd-acgt  plot
evo_ape tum  gt.txt  lsd_pose_acgt.txt -vas  --plot_mode xz --save_results lsd_acgt_APE.zip   -as  --save_plot ./lsd_acgt_ape
#lsd
evo_ape tum  gt.txt   lsd_pose.txt -vas  --plot_mode xz --save_results lsd_APE.zip   -as  --save_plot ./lsd_ape


# rpe
evo_rpe tum gt.txt lsd_pose_acgt.txt -r full -va --plot_mode xyz --save_plot ./lsd_acgt_rpe --save_results ./lsd_acgt_rpe.zip  -as

evo_rpe tum gt.txt lsd_pose.txt -r full -va  --plot_mode xyz --save_plot ./lsd_rpe --save_results ./lsd_rpe.zip  -as
