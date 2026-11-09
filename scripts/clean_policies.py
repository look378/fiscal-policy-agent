# -*- coding: utf-8 -*-
"""清洗抓取的政策原文：去掉网页导航噪声、按「第X条」切分、统一格式。

用法: python scripts/clean_policies.py [--src 目录] [--dst 目录]
输入: data/policies/*.txt
输出: data/policies_clean/*.md
"""
import argparse
import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 常见网页噪声（含中国政府网/地方财政局站点 chrome）
NOISE_PATTERNS = [
    r'^首页$', r'^EN$', r'^登录$', r'^个人中心$', r'^退出$', r'^邮箱$', r'^无障碍$',
    r'^字号[：:]?$', r'^默认$', r'^超大$', r'^打印$', r'^-->$', r'^网站无障碍开关$',
    r'^国务院公报$', r'^\d{4}年第\d+号$', r'^扫一扫在手机打开当前页$', r'^相关稿件$',
    r'^责任编辑', r'^【?打印本页】?$', r'^分享$', r'^关闭$', r'^返回顶部$',
    r'^https?://', r'^索引号', r'^主题分类', r'^发文机关', r'^成文日期', r'^发文字号',
    r'^发布日期', r'^有效性', r'^来源[：:]', r'^浏览量',
    # 实测残留（05 直达资金通知）
    r'^长者助手$', r'^我的收藏$', r'^规章库$', r'^收藏$', r'^政府信息公开$',
    r'^当前位置[：:]', r'^字体[：:]', r'^分享到', r'^微信扫一扫', r'^首页\s*>',
    r'^[^\s]{2,8}市[^\s]{0,6}(财政局|人民政府)$',
    # 页脚/备案
    r'版权所有', r'ICP备', r'^网站标识码', r'^主办单位', r'^承办单位', r'^网络支持',
    r'^京公网安备', r'^粤公网安备', r'^政府网站找错', r'^中国政府网',
]

RE_ARTICLE = re.compile(r'^(第[一二三四五六七八九十百零〇0-9]+条)')
# 没有「第X条」的公文（通知类）：用这些正文锚点定位起点
RE_BODY_ANCHOR = re.compile(r'^(现将|现就|现印发|为.*?(通知|印发).*?如下|各省、自治区、直辖市|'
                            r'党中央有关部门|国务院各部委|有关人民团体|各计划单列市)')
RE_END_NOISE = re.compile(r'^(附件[:：]?\d*|抄送[：:]|.{0,20}年.{1,2}月.{1,2}日$)')
# 站点页脚：命中即认为正文结束（这些内容出现在正文之后，且可能夹带重复的标题/前言）
RE_FOOTER = re.compile(
    r'(ICP备|公网安备|网站标识码|主办单位|承办单位|版权所有|网络支持|IPv6|网站纠错|'
    r'运行维护单位|中文域名|中国政府网\.政务|^注册$|^登录$|^退出$|^邮箱$|^发布$|^联系方式[：:]|'
    r'网站声明|联系我们|网站地图|关于我们|使用帮助|无障碍|长者助手|我的收藏|规章库|^收藏$|^字体|^打印$|^分享$)')


def clean_line(ln: str) -> str:
    ln = ln.strip().replace('\u3000', ' ').replace('\xa0', ' ')
    ln = ln.replace('\ufeff', '')
    ln = re.sub(r'\s+', ' ', ln)
    return ln


def find_body_start(lines):
    """优先找「第X条」；找不到则用公文正文锚点；都没有则返回 0。

    注意：不能用「附件」当正文起点——附件清单常出现在正文前言之前，会把正文截掉。
    """
    for i, ln in enumerate(lines):
        if RE_ARTICLE.match(ln):
            return i
    for i, ln in enumerate(lines):
        if RE_BODY_ANCHOR.match(ln):
            return i
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(ROOT, 'data', 'policies'))
    ap.add_argument('--dst', default=os.path.join(ROOT, 'data', 'policies_clean'))
    args = ap.parse_args()
    src, dst = args.src, args.dst
    if not os.path.isdir(src):
        raise SystemExit('源目录不存在: %s' % src)
    os.makedirs(dst, exist_ok=True)

    total_articles = 0
    for name in sorted(os.listdir(src)):
        if not name.endswith('.txt'):
            continue
        path = os.path.join(src, name)
        try:
            with io.open(path, encoding='utf-8-sig') as f:
                raw = f.read()
        except UnicodeDecodeError:
            with io.open(path, encoding='gbk', errors='replace') as f:
                raw = f.read()
            print('    [!] %s 非 UTF-8，已按 GBK 回退读取' % name)
        lines = [clean_line(x) for x in raw.splitlines()]
        lines = [x for x in lines if x and not any(re.match(p, x) for p in NOISE_PATTERNS)]

        start = find_body_start(lines)
        body = lines[start:]
        # 在页脚处截断：避免备案号/主办单位以及页脚后重复的标题与前言混进正文
        for i, ln in enumerate(body):
            if RE_FOOTER.search(ln):
                body = body[:i]
                break
        # 只删末尾 1~2 行"单位+日期/抄送/附件"尾巴（避免误删正文最后一句）
        for _ in range(2):
            if not body:
                break
            last = body[-1].strip()
            if re.match(r'^(附件[:：]?\d*|抄送[：:])', last) or \
               re.match(r'^[\u4e00-\u9fa5A-Za-z（）()]{2,20}\s*\d{4}年\d{1,2}月\d{1,2}日$', last):
                body.pop()
            else:
                break

        out = []
        for ln in body:
            m = RE_ARTICLE.match(ln)
            if m:
                out.append('')
                out.append('### ' + m.group(1))
                rest = ln[m.end():].strip()
                if rest:
                    out.append(rest)
            else:
                out.append(ln)

        text = '\n'.join(out)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        title = os.path.splitext(name)[0]
        content = '# %s\n\n> 来源：官方公开政策文本（已清洗，供知识库构建使用）\n%s\n' % (title, text)
        with io.open(os.path.join(dst, title + '.md'), 'w', encoding='utf-8') as f:
            f.write(content)
        n_articles = len(re.findall(r'^### 第', text, flags=re.M))
        total_articles += n_articles
        print('%-46s 条数=%-3d 字符=%d' % (name, n_articles, len(text)))
    print('[*] 合计条款: %d' % total_articles)


if __name__ == '__main__':
    main()
