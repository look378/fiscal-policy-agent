# -*- coding: utf-8 -*-
"""LLM 异常预警服务：Prometheus 拉指标 → 规则判定（去抖）→ LLM 归因（异步）→ 落盘/查询。

可靠性设计：
- 三态区分：查询失败(PromUnavailable) vs 无数据(None) vs 有值；**无数据不会被当成 0**
- 去抖：同一规则需连续 N 次命中（默认 2）；任何未评估该规则的轮次都会清零 streak
- 状态持久化：启动回放 alerts.jsonl（按每条告警的 ts 恢复冷却；遗留 pending 标记为 orphaned）
- 写盘隔离：单条写失败不影响整轮检查；文件写使用独立锁，避免阻塞 /livez /metrics
- 归因异步：告警先落盘，LLM 归因走队列；失败置 status=failed + error，并同步落盘
- 存活/就绪分离：`/livez` 只判进程；`/health` 反映 Prometheus 与检查是否停滞
- 指标：checks/fired/recent/pending/write_fail/diag_fail + last_check_timestamp

注意：本服务只负责"发现 + 归因 + 落盘/查询"，**不负责推送通知**（本地无 Alertmanager/Webhook）。
Prometheus 的 rule_files 规则仅在 Prometheus UI / API 可见，属已知限制。

用法:
    python alert_service.py --host 127.0.0.1 --port 8902 --interval 60
接口:
    GET /livez, GET /health, GET /alerts?limit=20, GET /metrics
"""
import argparse
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from collections import deque

import requests
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse

PROM = os.environ.get('PROM_URL', 'http://localhost:19090')
OLLAMA = os.environ.get('OLLAMA_BASE', 'http://localhost:11434')
MODEL = os.environ.get('OLLAMA_MODEL', 'deepseek-r1:7b')
DATA_DIR = os.environ.get('ALERT_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
ALERT_FILE = os.path.join(DATA_DIR, 'alerts.jsonl')

MIN_SAMPLES = int(os.environ.get('MIN_SAMPLES', '5'))
DIAG_TIMEOUT = int(os.environ.get('DIAG_TIMEOUT', '90'))
STALE_SECONDS = int(os.environ.get('STALE_SECONDS', '300'))
CONSECUTIVE = int(os.environ.get('CONSECUTIVE', '2'))
MAX_ALERTS = int(os.environ.get('MAX_ALERTS', '500'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('alert_service')

RULES = {
    'qa_service_down': {
        'metric': 'qa_service_up',
        'trigger': lambda v: v is not None and v == 0,     # 只在明确 up==0 时触发；None 不算
        'level': 'CRITICAL',
        'desc': '问答服务不可用（Prometheus 抓取不到 /metrics）',
        'cooldown': 300,
        'need_samples': False,
    },
    'high_error_rate': {
        'metric': 'error_rate',
        'trigger': lambda v: v is not None and v > 0.10,
        'level': 'HIGH',
        'desc': '问答服务错误率超过 10%',
        'cooldown': 300,
        'need_samples': True,
    },
    'high_refusal_rate': {
        'metric': 'refusal_rate',
        'trigger': lambda v: v is not None and v > 0.40,
        'level': 'MEDIUM',
        'desc': '知识库拒答率超过 40%（可能知识覆盖不足或检索失效）',
        'cooldown': 600,
        'need_samples': True,
    },
    'high_latency': {
        'metric': 'avg_latency_ms',
        'trigger': lambda v: v is not None and v > 20000,
        'level': 'MEDIUM',
        'desc': '平均响应延迟超过 20 秒',
        'cooldown': 300,
        'need_samples': True,
    },
}

QUERIES = {
    'requests_rate': 'sum(rate(fiscal_qa_requests_total[1m]))',
    'sample_size': 'sum(increase(fiscal_qa_requests_total[5m]))',
    'error_rate': 'sum(rate(fiscal_qa_errors_total[5m])) / clamp_min(sum(rate(fiscal_qa_requests_total[5m])), 0.0001)',
    'refusal_rate': 'sum(rate(fiscal_qa_refused_total[5m])) / clamp_min(sum(rate(fiscal_qa_requests_total[5m])), 0.0001)',
    'avg_latency_ms': 'sum(rate(fiscal_qa_latency_ms_total[5m])) / clamp_min(sum(rate(fiscal_qa_requests_total[5m])), 0.0001)',
    'qa_service_up': 'up{job="fiscal-qa-service"}',
}

STATE = {
    'last_fire': {}, 'alerts': deque(maxlen=MAX_ALERTS), 'checks': 0, 'fired': 0,
    'last_check_ts': 0, 'prom_ok': False, 'history': deque(maxlen=500),
    'streak': {}, 'write_fail': 0, 'diag_fail': 0, 'total_alerts': 0,
}
LOCK = threading.Lock()          # 保护 STATE
WRITE_LOCK = threading.Lock()    # 只保护文件写，避免写盘阻塞指标接口

app = FastAPI(title='LLM 异常预警服务')


class PromUnavailable(RuntimeError):
    pass


def prom_query(expr):
    """查询 Prometheus；'无数据'返回 None，'查询失败'抛 PromUnavailable（三态区分）。"""
    try:
        r = requests.get(PROM + '/api/v1/query', params={'query': expr}, timeout=10)
        r.raise_for_status()
        res = r.json().get('data', {}).get('result', [])
        if not res:
            return None
        return float(res[0]['value'][1])
    except Exception as e:                          # noqa: BLE001
        raise PromUnavailable(str(e)[:200])


def snapshot():
    vals = {k: prom_query(expr) for k, expr in QUERIES.items()}
    return {
        'qps': round(vals.get('requests_rate') or 0, 3),
        'sample_size': int(vals.get('sample_size') or 0),
        'error_rate': round(vals.get('error_rate') or 0, 4),
        'refusal_rate': round(vals.get('refusal_rate') or 0, 4),
        'avg_latency_ms': round(vals.get('avg_latency_ms') or 0, 1),
        # 保留 None，不折叠成 0（折叠会把"指标缺失"误判成"服务 down"）
        'qa_service_up': vals.get('qa_service_up'),
        'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def llm_diagnose(rule_key, rule, snap):
    """调用本地 LLM 生成异常原因与处置建议；返回 (ok, text)。"""
    prompt = (
        '你是 AI 系统运维专家。下面是「财政政策问答智能体」的监控指标与触发的告警规则。\n'
        '指标快照：%s\n'
        '告警：%s（%s），级别 %s，当前值 %s\n'
        '请用中文输出三行，不要多余内容：\n'
        '可能原因：...\n'
        '处置建议：...\n'
        '是否需要人工介入：是/否'
    ) % (json.dumps(snap, ensure_ascii=False), rule_key, rule['desc'], rule['level'],
         snap.get(rule['metric']))
    for attempt in range(1, 3):
        try:
            r = requests.post(OLLAMA + '/api/chat', json={
                'model': MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False,
                'options': {'temperature': 0.2, 'num_predict': 900, 'num_ctx': 4096},
            }, timeout=DIAG_TIMEOUT)
            r.raise_for_status()
            txt = (r.json().get('message') or {}).get('content', '')
            keep = []
            for line in txt.splitlines():
                s = re.sub(r'^\s*(?:[-*#]+\s*|\d+[.、)]\s*)', '', line).strip().replace('**', '')
                if s.startswith(('可能原因', '处置建议', '是否需要')):
                    keep.append(s)
            return True, ('\n'.join(keep[:3]) if keep else txt.strip()[:400])
        except Exception as e:                      # noqa: BLE001
            log.warning('LLM 归因失败（第 %d 次）: %s', attempt, str(e)[:150])
            time.sleep(2)
    return False, '（LLM 归因失败，可稍后在 Ollama 恢复后重试）'


def _set_alert_state(alert_id, status, diagnosis):
    """原子更新内存中的告警状态。"""
    with LOCK:
        for item in STATE['alerts']:
            if item.get('id') == alert_id:
                item['status'] = status
                item['diagnosis'] = diagnosis
                return


def diag_worker():
    """后台归因线程：避免 LLM 调用阻塞指标采样。"""
    while True:
        alert = DIAG_QUEUE.get()
        try:
            ok, diag = llm_diagnose(alert['rule'], RULES[alert['rule']], alert['snapshot'])
            status = 'diagnosed' if ok else 'failed'
            _set_alert_state(alert['id'], status, diag)
            if not _append({'id': alert['id'], 'status': status, 'diagnosis': diag,
                            'error': None if ok else diag}, kind='diagnosis'):
                log.warning('归因结果落盘失败（id=%s），内存状态=%s', alert['id'], status)
        except Exception:                           # noqa: BLE001
            log.exception('diag worker 异常')
            with LOCK:
                STATE['diag_fail'] += 1
            _set_alert_state(alert['id'], 'failed', '（归因线程异常）')
            _append({'id': alert['id'], 'status': 'failed', 'diagnosis': '（归因线程异常）'},
                    kind='diagnosis')
        finally:
            DIAG_QUEUE.task_done()


def _append(obj, kind='alert'):
    """追加写 JSONL；失败不抛（返回 False）并计数。文件写用独立锁，不阻塞 /metrics。"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with WRITE_LOCK:
            with open(ALERT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                f.flush()
        return True
    except Exception as e:                          # noqa: BLE001
        log.error('写告警文件失败（%s）: %s', kind, e)
        with LOCK:
            STATE['write_fail'] += 1
        return False


def emit(rule_key, rule, snap):
    now = time.time()
    alert = {
        'id': '%s-%d-%s' % (rule_key, int(now * 1000), uuid.uuid4().hex[:6]),
        'rule': rule_key, 'level': rule['level'], 'desc': rule['desc'],
        'snapshot': snap, 'diagnosis': '（LLM 归因中）', 'status': 'pending',
        'ts': round(now, 3),                     # epoch 秒：回放时用于恢复各规则真实冷却
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if not _append(alert):
        alert['status'] = 'persist_failed'
    with LOCK:
        STATE['alerts'].append(alert)
        STATE['fired'] += 1
        STATE['total_alerts'] += 1
        STATE['last_fire'][rule_key] = now
        STATE['history'].append(now)
    log.warning('[ALERT][%s] %s | snapshot=%s', rule['level'], rule['desc'],
                json.dumps(snap, ensure_ascii=False))
    try:
        DIAG_QUEUE.put_nowait(alert)
    except queue.Full:
        alert['status'] = 'failed'
        alert['diagnosis'] = '（归因队列已满，丢弃本次 LLM 归因）'
        with LOCK:
            STATE['diag_fail'] += 1
        _append({'id': alert['id'], 'status': 'failed', 'diagnosis': alert['diagnosis']},
                kind='diagnosis')


def replay_history():
    """启动回放：恢复最近告警、按每条告警 ts 恢复冷却；遗留 pending 标记为 orphaned。"""
    if not os.path.exists(ALERT_FILE):
        return
    alerts = {}
    try:
        with open(ALERT_FILE, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue                     # 容忍崩溃留下的半行
                aid = obj.get('id')
                if not aid:
                    continue
                if 'rule' in obj:
                    alerts.setdefault(aid, obj)
                elif aid in alerts:
                    alerts[aid].update({k: v for k, v in obj.items() if k != 'id'})
    except Exception as e:                       # noqa: BLE001
        log.warning('回放告警历史失败: %s', e)
        return

    now = time.time()
    recent = list(alerts.values())[-MAX_ALERTS:]
    orphaned = 0
    with LOCK:
        for a in recent:
            if a.get('status') == 'pending':
                # 上一进程在归因完成前退出：不再有 worker 会处理它 → 标记 orphaned 并落盘
                a['status'] = 'orphaned'
                a['diagnosis'] = '（进程重启前未完成归因）'
                orphaned += 1
            STATE['alerts'].append(a)
        STATE['total_alerts'] = len(alerts)
        for a in alerts.values():
            rule = a.get('rule')
            ts = a.get('ts')
            if rule and isinstance(ts, (int, float)):
                # clamp 到 now，避免时钟回拨导致冷却被无限期延长
                ts = min(float(ts), now)
                STATE['last_fire'][rule] = max(STATE['last_fire'].get(rule, 0), ts)
                if now - ts <= 3600:
                    STATE['history'].append(ts)
    for a in recent:
        if a.get('status') == 'orphaned':
            _append({'id': a['id'], 'status': 'orphaned',
                     'diagnosis': a['diagnosis']}, kind='diagnosis')
    log.info('已回放 %d 条历史告警（其中 %d 条 pending 标记为 orphaned）', len(alerts), orphaned)


def check_loop(interval):
    while True:
        try:
            snap = snapshot()
            with LOCK:
                STATE['checks'] += 1
                STATE['last_check_ts'] = time.time()
                STATE['prom_ok'] = True
            for key, rule in RULES.items():
                # 任何"本轮没有评估该规则"的情况都要清零 streak，避免跨中断累计
                if rule.get('need_samples') and snap['sample_size'] < MIN_SAMPLES:
                    with LOCK:
                        STATE['streak'][key] = 0
                    continue
                try:
                    val = snap.get(rule['metric'])
                    if val is None and key == 'qa_service_down':
                        log.warning('up 指标无数据（target 缺失/重载中），本轮不判 down')
                        with LOCK:
                            STATE['streak'][key] = 0
                        continue
                    hit = rule['trigger'](val)
                except Exception as e:        # noqa: BLE001
                    log.exception('规则 %s 判定异常: %s', key, e)
                    with LOCK:
                        STATE['streak'][key] = 0
                    continue
                with LOCK:
                    streak = STATE['streak'].get(key, 0) + 1 if hit else 0
                    STATE['streak'][key] = streak
                    last = STATE['last_fire'].get(key, 0)
                if hit and streak >= CONSECUTIVE and (time.time() - last) > rule['cooldown']:
                    emit(key, rule, snap)
        except PromUnavailable as e:
            with LOCK:
                STATE['prom_ok'] = False
                STATE['streak'] = {}          # Prometheus 不可达：所有规则 streak 作废
            log.warning('Prometheus 不可达，本轮跳过（不误报服务 down）: %s', e)
        except Exception as e:                    # noqa: BLE001
            log.exception('check loop 异常: %s', e)
            with LOCK:
                STATE['streak'] = {}
        time.sleep(interval)


@app.get('/livez')
def livez():
    """存活检查：只证明进程在跑，不查 Prometheus（避免下游故障导致 liveness 重启）。"""
    return {'status': 'alive'}


@app.get('/health')
def health():
    """就绪检查：Prometheus 可达且检查未停滞才算就绪（readiness 用）。"""
    with LOCK:
        last = STATE['last_check_ts']
        prom_ok = STATE['prom_ok']
        checks = STATE['checks']
    stale = (time.time() - last) > STALE_SECONDS if last else True
    ok = prom_ok and not stale
    body = {'status': 'ok' if ok else 'degraded', 'prometheus_reachable': prom_ok,
            'seconds_since_last_check': int(time.time() - last) if last else None,
            'checks': checks}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get('/alerts')
def alerts(limit: int = Query(default=20, ge=1, le=200)):
    with LOCK:
        items = [dict(a) for a in list(STATE['alerts'])[-limit:][::-1]]
        total = STATE['total_alerts']
    return {'total': total, 'returned': len(items), 'alerts': items}


@app.get('/metrics', response_class=PlainTextResponse)
def metrics():
    now = time.time()
    with LOCK:
        checks, fired = STATE['checks'], STATE['fired']
        last_check = STATE['last_check_ts']
        recent = len([t for t in STATE['history'] if now - t <= 3600])
        pending = len([a for a in STATE['alerts'] if a.get('status') == 'pending'])
        write_fail, diag_fail = STATE['write_fail'], STATE['diag_fail']
    lines = [
        '# HELP fiscal_alert_checks_total 预警检查次数', '# TYPE fiscal_alert_checks_total counter',
        'fiscal_alert_checks_total %d' % checks,
        '# HELP fiscal_alert_fired_total 触发告警次数', '# TYPE fiscal_alert_fired_total counter',
        'fiscal_alert_fired_total %d' % fired,
        '# HELP fiscal_alert_recent_total 最近 1 小时告警数（含回放）', '# TYPE fiscal_alert_recent_total gauge',
        'fiscal_alert_recent_total %d' % recent,
        '# HELP fiscal_alert_pending_diagnosis 待 LLM 归因的告警数', '# TYPE fiscal_alert_pending_diagnosis gauge',
        'fiscal_alert_pending_diagnosis %d' % pending,
        '# HELP fiscal_alert_write_fail_total 告警落盘失败次数', '# TYPE fiscal_alert_write_fail_total counter',
        'fiscal_alert_write_fail_total %d' % write_fail,
        '# HELP fiscal_alert_diag_fail_total 归因失败/丢弃次数', '# TYPE fiscal_alert_diag_fail_total counter',
        'fiscal_alert_diag_fail_total %d' % diag_fail,
        '# HELP fiscal_alert_last_check_timestamp_seconds 最近一次成功检查的时间戳', '# TYPE fiscal_alert_last_check_timestamp_seconds gauge',
        'fiscal_alert_last_check_timestamp_seconds %.3f' % (last_check or 0),
    ]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8902')))
    ap.add_argument('--interval', type=int, default=60)
    # 默认只监听本机；容器/K8s 部署时传 --host 0.0.0.0
    ap.add_argument('--host', default=os.environ.get('HOST', '127.0.0.1'))
    args = ap.parse_args()
    if args.interval < 5:
        raise SystemExit('--interval 必须 >= 5 秒（否则会忙循环）')
    if args.interval * 2 > STALE_SECONDS:
        raise SystemExit('--interval 过大：需满足 interval*2 <= STALE_SECONDS(%d)，否则 /health 永远 503'
                         % STALE_SECONDS)
    if CONSECUTIVE < 1:
        raise SystemExit('CONSECUTIVE 必须 >= 1（0 等于关闭去抖）')
    if MAX_ALERTS < 1:
        raise SystemExit('MAX_ALERTS 必须 >= 1')
    if MIN_SAMPLES < 0:
        raise SystemExit('MIN_SAMPLES 必须 >= 0')
    os.makedirs(DATA_DIR, exist_ok=True)
    replay_history()
    threading.Thread(target=diag_worker, daemon=True).start()
    threading.Thread(target=check_loop, args=(args.interval,), daemon=True).start()
    log.info('预警服务启动: interval=%ds prometheus=%s model=%s min_samples=%d consecutive=%d',
             args.interval, PROM, MODEL, MIN_SAMPLES, CONSECUTIVE)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
