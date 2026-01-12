# Progress Log: RULER 非 Docker 兼容性探索

## Session: 2026-01-13

### 已完成操作

| 时间 | 操作 | 结果 |
|------|------|------|
| - | 读取 `scripts/run_ruler_docker.sh` | 了解 Docker 配置 |
| - | 读取 `eval/RULER/requirements.txt` | 获取依赖列表 |
| - | 读取 `eval/RULER/Dockerfile` | 了解基础镜像 |
| - | 检查 conda 环境包版本 | PyTorch 2.9.1, transformers 4.51.0 |
| - | 测试 flash_attn 导入 | ✅ 成功 |
| - | 测试 flashinfer 导入 | ✅ 成功 |
| - | 测试 vllm 导入 | ✅ 成功 |
| - | 测试 NanoVLLMModel 导入 | ✅ 成功 |
| - | 测试 call_api.py 导入 | ❌ huggingface_hub 冲突 |
| - | 检查 Docker 镜像包版本 | huggingface_hub 0.23.4 |
| - | 对比分析 | 完成问题定位 |

### 遇到的错误

| 错误 | 文件 | 原因 | 解决状态 |
|------|------|------|----------|
| `ImportError: ModelFilter` | call_api.py | huggingface_hub 版本过新 | 已记录解决方案 |

### 文件操作

| 操作 | 文件 |
|------|------|
| 读取 | `scripts/run_ruler_docker.sh` |
| 读取 | `eval/RULER/requirements.txt` |
| 读取 | `eval/RULER/scripts/requirements.txt` |
| 读取 | `eval/RULER/Dockerfile` |
| 读取 | `eval/RULER/scripts/run.sh` |
| 创建 | `.claude/rules/planning-with-files.md` |
| 创建 | `task_plan.md` |
| 创建 | `findings.md` |
| 创建 | `progress.md` |

### 结论

**核心问题已定位**: huggingface_hub 版本冲突是导致 RULER 无法在当前 conda 环境运行的主要原因。

**推荐方案**: 继续使用 Docker，因为：
1. 版本兼容性已验证
2. 不影响现有开发环境
3. 之前的 RULER 测试已在 Docker 中成功运行

### 下一步行动

- [ ] ~~如需脱离 Docker，执行版本降级~~ (不推荐)
- [ ] 创建隔离 conda 环境测试 NeMo 2.6.1
- [x] 继续使用 Docker 运行 RULER（当前方案）
- [ ] 验证 NeMo 2.6.1 与 RULER 代码兼容性

---

## Session 2: 2026-01-13 (续)

### NeMo 最新版本调研

| 时间 | 操作 | 结果 |
|------|------|------|
| - | 搜索 NeMo ModelFilter issue | 发现 #10272 已于 2024-09 关闭 |
| - | 搜索 NeMo transformers 兼容性 | 4.41.0 有问题，4.51+ 可用 |
| - | 查看 NeMo PyPI 最新版本 | 2.6.1 (2026-01-09 发布) |
| - | pip dry-run 检查依赖 | huggingface_hub>=0.24 ✅ |
| - | 分析 NeMo 2.x 变化 | LLM/VLM 将迁移到新仓库 |

### 关键发现

| 发现 | 影响 |
|------|------|
| ModelFilter 已从 NeMo 2.x 移除 | huggingface_hub 版本冲突可解决 |
| NeMo 2.6.1 要求 huggingface_hub>=0.24 | 当前 0.36.0 兼容 ✅ |
| transformers 无严格版本限制 | 当前 4.51.0 可能兼容 |
| NeMo 1.x → 2.x 是重大升级 | RULER 代码可能需要适配 |

### 更新后的方案评估

| 方案 | 可行性 | 风险 | 推荐度 |
|------|--------|------|--------|
| A: 降级包版本 | 高 | 高 (破坏其他依赖) | ⭐ |
| B: 专用 conda 环境 | 高 | 低 | ⭐⭐⭐⭐ |
| C: 继续 Docker | 高 | 无 | ⭐⭐⭐⭐⭐ |
| D: 升级 NeMo 2.6.1 | 中 | 中 (需验证 API) | ⭐⭐⭐ |

---

## Session 3: 2026-01-13 (固定版本库约束分析)

### 固定版本库调研

| 时间 | 操作 | 结果 |
|------|------|------|
| - | 检查固定版本库版本 | torch 2.9.1, vllm 0.12.1, flashinfer 0.5.3, flash_attn 2.8.3, sglang 0.5.6 |
| - | 检查 vllm transformers 要求 | `>=4.56.0, <5` |
| - | 检查 sglang transformers 要求 | `==4.57.1` (严格!) |
| - | 检查 NeMo ASR transformers 要求 | `~=4.53.0` |
| - | 分析版本冲突 | 无法同时满足 sglang 和 NeMo ASR |
| - | 分析 RULER NeMo 依赖 | 仅使用 manifest_utils (JSONL 工具) |

### 关键发现

| 发现 | 影响 |
|------|------|
| sglang 严格要求 transformers==4.57.1 | 与 NeMo ASR ~=4.53.0 不兼容 |
| 当前 transformers 4.51.0 已有冲突 | vllm/sglang 要求未满足但能运行 |
| RULER 仅用 NeMo 的 JSONL 工具 | 可以 10 行代码本地替代 |
| NeMo base 无 transformers 要求 | 但 ASR/NLP extra 需要 4.53.x |

### 最终方案选择

**方案 A: 本地实现替代 NeMo 依赖** ⭐⭐⭐⭐⭐

理由:
1. RULER 仅使用 NeMo 的简单 JSONL 工具
2. 可用 10 行代码完全替代
3. 零版本冲突
4. 保持所有固定版本库不变

### 执行计划

1. [ ] 创建 `eval/RULER/scripts/utils/manifest_utils.py`
2. [ ] 修改 8 个 RULER 文件的 import 路径
3. [ ] 克隆/创建 conda 环境
4. [ ] 验证 RULER 运行
