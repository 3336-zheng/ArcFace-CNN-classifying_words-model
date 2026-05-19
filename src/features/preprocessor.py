"""公共数据预处理 - JSONL 数据加载、指令映射创建"""

import json
import os
from collections import Counter


class DataPreprocessor:
    """加载 JSONL 数据并创建指令映射"""

    def __init__(self):
        self.commands_counter = Counter()
        self.valid_data = []
        self.command_mapping = {}
        self.reverse_mapping = {}

    def load_and_analyze_data(self, jsonl_path):
        """加载 JSONL 数据并分析分布"""
        print("正在加载和分析数据...")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    audio_path = item["audio"]["path"]
                    sentence = item["sentence"]

                    if os.path.exists(audio_path):
                        self.valid_data.append({
                            "audio_path": audio_path,
                            "sentence": sentence,
                            "duration": item.get("duration", 0)
                        })
                        self.commands_counter[sentence] += 1
                    else:
                        print(f"警告: 文件不存在 - {audio_path}")

                except json.JSONDecodeError as e:
                    print(f"JSON解析错误 (行 {line_num}): {e}")
                except KeyError as e:
                    print(f"键错误 (行 {line_num}): {e}")

        print(f"\n数据统计:")
        print(f"总有效样本: {len(self.valid_data)}")
        print(f"指令类别数: {len(self.commands_counter)}")
        print("\n指令分布:")
        for cmd, count in self.commands_counter.most_common():
            print(f"  {cmd}: {count} 样本")

        return self.valid_data

    def create_command_mapping(self, min_samples=10):
        """创建指令映射，过滤样本太少的类别"""
        valid_commands = [
            cmd for cmd, count in self.commands_counter.items()
            if count >= min_samples
        ]
        valid_commands.sort()

        self.command_mapping = {cmd: idx for idx, cmd in enumerate(valid_commands)}
        self.reverse_mapping = {idx: cmd for cmd, idx in self.command_mapping.items()}

        print(f"\n使用的指令类别 ({len(valid_commands)} 个):")
        for cmd in valid_commands:
            print(f"  {cmd} -> {self.command_mapping[cmd]}")

        return self.command_mapping

    def save_command_mapping_json(self, filename="command_mapping.json"):
        """保存指令映射为 JSON 格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.command_mapping, f, ensure_ascii=False, indent=2)
        print(f"✓ 指令映射已保存为 {filename}")

    def load_command_mapping_json(self, filename):
        """从 JSON 文件加载指令映射"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.command_mapping = json.load(f)
        self.reverse_mapping = {v: k for k, v in self.command_mapping.items()}
        return self.command_mapping
