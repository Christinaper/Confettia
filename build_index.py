#!/usr/bin/env python3
# build_index.py
# 独立脚本：构建 / 更新 Obsidian 向量索引
# 用法：
#   python build_index.py              # 增量更新
#   python build_index.py --force      # 强制全量重建
#   python build_index.py --stats      # 只查看当前索引统计

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def main():
    force      = "--force" in sys.argv
    stats_only = "--stats" in sys.argv

    vault = os.getenv("OBSIDIAN_VAULT", "")
    if not vault:
        print("❌ 请在 .env 中设置 OBSIDIAN_VAULT=/path/to/your/vault")
        sys.exit(1)

    if not Path(vault).exists():
        print(f"❌ 路径不存在：{vault}")
        sys.exit(1)

    from tools.rag import _get_collection, build_index

    if stats_only:
        col = _get_collection()
        print(f"📊 当前索引：{col.count()} 个 chunks")
        return

    print(f"🔍 扫描 Obsidian 库：{vault}")
    if force:
        print("⚠️  强制全量重建（将删除现有索引）")

    stats = build_index(vault, force=force)
    print(
        f"✅ 索引完成\n"
        f"   新增：{stats['added']} 篇\n"
        f"   更新：{stats['updated']} 篇\n"
        f"   跳过：{stats['skipped']} 篇（未修改）\n"
        f"   删除：{stats['deleted']} 条（文件已移除）"
    )

if __name__ == "__main__":
    main()
