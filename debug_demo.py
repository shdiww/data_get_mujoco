import os
import runpy
import shlex
import sys


# 使用 os.path.abspath 获取绝对路径，防止 Hydra 切换工作目录后导致相对路径失效
command = "python train.py --config-name=train_diffusion_unet_mujoco_image_workspace "

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
