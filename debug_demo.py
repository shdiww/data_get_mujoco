import os
import runpy
import shlex
import sys

# 直接在这里填写你想运行的命令行，便于一键 debug
# # 示例：command = "python eval.py -c ckpt.pt -o outputs/ -d cuda:0 --extra-flag 1"

# =============================================================================
# 1. 设置你的数据集路径 (相对路径或绝对路径均可)
#    请确保该目录下包含 videos 文件夹或 replay_buffer.zarr
# =============================================================================
my_dataset_path = "data/mujoco_data/data" 

# =============================================================================
# 2. 自动检查路径并生成命令
# =============================================================================
if not os.path.exists(my_dataset_path):
    print(f"\n[Error] 找不到数据集路径: {os.path.abspath(my_dataset_path)}")
    if os.path.exists("data"):
        print(f"data 目录下现有的文件夹: {os.listdir('data')}")
    print("请修改 debug_demo.py 中的 my_dataset_path 变量。\n")
else:
    print(f"[Info] 使用数据集路径: {os.path.abspath(my_dataset_path)}")
    # 简单的检查：看目录下是否有 videos 或 data 文件夹
    contents = os.listdir(my_dataset_path)
    print(f"[Info] 目录内容: {contents}")
    if 'videos' not in contents and 'data' not in contents and 'replay_buffer.zarr' not in contents:
        print("[Warning] ⚠️  在该目录下未找到 'videos' 或 'data' 文件夹。请确认路径层级是否正确。")
        print("           通常数据集目录应该包含一个 'videos' 文件夹。")

# 使用 os.path.abspath 获取绝对路径，防止 Hydra 切换工作目录后导致相对路径失效
command = f"python train.py --config-name=train_diffusion_unet_mujoco_image_workspace task.dataset_path='{os.path.abspath(my_dataset_path)}'"

debug_dir = ""


def main():
    # 解析命令行字符串
    args = shlex.split(command)
    # 允许以 python 开头，也可直接以脚本名开头
    if args and args[0] == "python":
        args = args[1:]
    if not args:
        raise ValueError("command 为空，请在文件顶部填写要运行的命令行")

    script = args[0]
    script_args = args[1:]

    # 切换到目标目录
    if debug_dir:
        os.chdir(debug_dir)

    # 设置 sys.argv 并运行脚本，效果等同于在命令行执行
    sys.argv = [script] + script_args
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
