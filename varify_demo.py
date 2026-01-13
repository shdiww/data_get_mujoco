import click
import zarr
import numpy as np

@click.command()
@click.option('-d', '--dataset_path', default='data/mujoco_demo/replay_buffer.zarr', help='Path to zarr dataset')
def check_stats(dataset_path):
    print(f"Loading {dataset_path}...")
    root = zarr.open(dataset_path, mode='r')
    
    # 获取所有 state 数据
    state_data = root['data']['state'][:] # Load all into memory
    print(f"State Data Shape: {state_data.shape}")
    
    # 计算统计量
    mean = np.mean(state_data, axis=0)
    min_val = np.min(state_data, axis=0)
    max_val = np.max(state_data, axis=0)
    std = np.std(state_data, axis=0)
    scale_estimated = 2.0 / (max_val - min_val)
    
    print("\n" + "="*60)
    print("Zarr Dataset Full Statistics (Computed Manually)")
    print("="*60)
    print(f"{'Idx':<3} | {'Name':<10} | {'Mean':<10} | {'Min':<10} | {'Max':<10} | {'Est. Scale':<10}")
    print("-" * 75)
    
    names = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'g']
    
    for i in range(7):
        name = names[i] if i < 7 else str(i)
        print(f"{i:<3} | {name:<10} | {mean[i]:<10.4f} | {min_val[i]:<10.4f} | {max_val[i]:<10.4f} | {scale_estimated[i]:<10.4f}")
        
    print("-" * 75)
    
    # 重点检查 Z 轴
    print("\n[Analysis]")
    print(f"Z-Axis Max is {max_val[2]:.4f}.")
    print(f"Your Test Episode Start Z is 0.488.")
    if 0.488 > max_val[2]:
        print(f"❌ WARNING: Test Z (0.488) is HIGHER than Training Max ({max_val[2]:.4f})!")
        print("   This confirms Distribution Shift causing the drop.")
    else:
        print("✅ Z-Axis seems within range.")

    # 重点检查 Rx
    print(f"\nRx-Axis Mean is {mean[3]:.4f}. (Your Normalizer says 0.757)")
    print(f"Rx-Axis Scale is {scale_estimated[3]:.4f}. (Your Normalizer says 0.318)")
    print("   (If these match the Normalizer stats, then the data is simply Euler and valid.)")

if __name__ == '__main__':
    check_stats()