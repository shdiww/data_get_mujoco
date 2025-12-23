#!/usr/bin/env python3
"""
增强版 inspect_zarr.py：
- 列出 zarr 数据集结构
- 支持查看某个数组的前 N 条（--show, --head）
- 支持将数组导出为 .npy 或 .csv（--save, --out, --format）

示例：
  列出： python inspect_zarr.py /path/to/replay_buffer.zarr --list
  查看： python inspect_zarr.py /path/to/replay_buffer.zarr --show action --head 5
  导出： python inspect_zarr.py /path/to/replay_buffer.zarr --show actions --save --out actions.csv --format csv
"""

import argparse
import os
import sys
import zarr
import numpy as np


def list_datasets(group, prefix=''):
    """递归列出 Group 下所有数组及其形状"""
    for key in sorted(group.keys()):
        item = group[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, zarr.Group):
            print(f"📂 {path}/")
            list_datasets(item, path)
        elif isinstance(item, zarr.Array):
            print(f"🔢 {path}: shape={item.shape}, dtype={item.dtype}")


def get_item(root, path):
    """根据路径取到对应的 Group/Array（path 可以是 'a/b/c' 或单个 key）"""
    if not path:
        return root
    parts = [p for p in path.split('/') if p]
    cur = root
    for p in parts:
        if isinstance(cur, zarr.Group):
            if p in cur:
                cur = cur[p]
            else:
                raise KeyError(f"Path not found: {path} (missing {p})")
        else:
            raise KeyError(f"Reached array before end of path at {p}")
    return cur


def find_candidates(root, name):
    """按名字模糊匹配，返回所有可能路径"""
    name = name.lower()
    matches = []

    def walk(g, prefix=''):
        for k in g.keys():
            item = g[k]
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(item, zarr.Group):
                walk(item, path)
            elif isinstance(item, zarr.Array):
                if name in k.lower():
                    matches.append(path)

    walk(root)
    return matches


def show_array(arr, head=10):
    """打印数组信息和前若干条样例"""
    print(f"shape={arr.shape}, dtype={arr.dtype}")
    try:
        data = arr[:head]
        # 有时是 bytes/object，需要尽量展示可读形式
        np.set_printoptions(threshold=1000)
        print(data)
    except Exception as e:
        print(f"无法读取数据: {e}")


def save_array(arr, outpath, fmt=None):
    """保存数组：npy 或 csv（仅支持 1D/2D）"""
    data = np.asarray(arr[:])
    if fmt is None or fmt == 'npy':
        if not outpath.endswith('.npy'):
            outpath = outpath + '.npy'
        np.save(outpath, data)
        print(f"Saved NPY: {outpath}")
        return outpath

    if fmt == 'csv':
        if data.ndim > 2:
            raise ValueError('CSV 导出只支持 1D 或 2D 数组')
        # 使用 numpy.savetxt，会把非数值按 str 写入
        np.savetxt(outpath, data.reshape(data.shape[0], -1) if data.ndim == 1 else data, delimiter=',', fmt='%s')
        print(f"Saved CSV: {outpath}")
        return outpath

    raise ValueError(f'Unsupported format: {fmt}')


def main():
    parser = argparse.ArgumentParser(description='Inspect and extract datasets from a Zarr store (支持列出/查看/导出)')
    parser.add_argument('zarr_path', help='Path to the Zarr file or directory')
    parser.add_argument('--list', action='store_true', help='List all arrays in the store')
    parser.add_argument('--show', help="Show a dataset by name or path (eg 'action' or 'data/action')")
    parser.add_argument('--head', type=int, default=10, help='Number of rows/items to show when using --show')
    parser.add_argument('--save', action='store_true', help='Save the displayed dataset to disk')
    parser.add_argument('--out', help='Output file path (use .csv for CSV or omit extension for .npy)')
    parser.add_argument('--format', choices=['npy', 'csv'], help='Output format when using --save')

    args = parser.parse_args()

    if not os.path.exists(args.zarr_path):
        print(f"Path not found: {args.zarr_path}")
        sys.exit(1)

    try:
        root = zarr.open(args.zarr_path, mode='r')
    except Exception as e:
        print(f"Error opening Zarr store: {e}")
        sys.exit(1)

    if args.list or not args.show:
        # 如果是仅 --list 或 没有指定 --show，则列出所有数组
        print(f"Inspecting Zarr dataset: {args.zarr_path}\n")
        list_datasets(root)
        if not args.show:
            return

    if args.show:
        # 尝试按名字模糊匹配，优先使用精确路径，如果有多条匹配则列出来让用户选择
        try:
            # 先尝试直接以路径访问
            try:
                item = get_item(root, args.show)
                path_used = args.show
            except KeyError:
                # 找到模糊匹配
                candidates = find_candidates(root, args.show)
                if not candidates:
                    print(f"No dataset matched: {args.show}")
                    return
                if len(candidates) > 1:
                    print('Multiple matches found:')
                    for c in candidates:
                        print('  -', c)
                    print("Please re-run with the full path of the desired dataset.")
                    return
                path_used = candidates[0]
                item = get_item(root, path_used)

            if isinstance(item, zarr.Group):
                print(f"{path_used} is a Group. Listing contents:")
                list_datasets(item, prefix=path_used)
                return

            print(f"Dataset: {path_used}")
            show_array(item, head=args.head)

            if args.save:
                if not args.out:
                    print('Please specify --out when using --save')
                    return
                saved = save_array(item, args.out, fmt=args.format)
                print('Saved to', saved)

        except Exception as e:
            print(f"Error handling dataset: {e}")


if __name__ == '__main__':
    main()