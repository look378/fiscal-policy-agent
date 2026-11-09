# -*- coding: utf-8 -*-
"""统一的拒答判定工具（服务端与评测脚本共用，避免多处定义分叉）。

分两层：
- STRONG_REFUSAL_MARKERS：**明确表示"知识库里没有"**，服务端用它判定"文本拒答"；
- GUIDE_MARKERS：引导用户去官方渠道的提示语。它们可能出现在**正常回答的补充说明里**，
  单独出现不能算拒答（例如"……以当地最新规定为准"），只供评测负样本判定参考。
"""

import re

# 引用编号（与 serving/qa_service.py 的 parse_citations 同口径）：
# 支持 [1] / 【1】 / [1,2] / [1、2] / [1-3]；政策文本里的 （1） 是列表编号，不算引用
CITE_PATTERN = re.compile(r'[\[【]\s*([0-9\s,，、\-~至到]+?)\s*[\]】]')


def has_citation(text):
    """答案里是否出现编号引用（用于"负样本是否真拒答"的判定）。"""
    return bool(CITE_PATTERN.search(text or ''))


STRONG_REFUSAL_MARKERS = (
    '没有查到', '未查到', '未找到', '无法回答', '无法依据', '没有相关规定',
    '知识库中没有', '知识库中未', '不在本办法', '不在本库',
    '未收录', '没有收录', '未包含', '无法直接回答', '无法给出',
)

GUIDE_MARKERS = (
    '需要查阅', '需查阅', '建议查阅', '建议您查阅', '咨询当地', '财政部官网',
    '以当地最新', '以地方财政部门', '以最新文件为准', '以财政部最新',
    '无法确定', '不能确定',
)

# 评测负样本用：只要出现任一（且无编号引用）即视为"正确拒答"
REFUSAL_MARKERS = STRONG_REFUSAL_MARKERS + GUIDE_MARKERS

__all__ = ['STRONG_REFUSAL_MARKERS', 'GUIDE_MARKERS', 'REFUSAL_MARKERS', 'has_citation']
