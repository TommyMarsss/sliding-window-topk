package topk

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// FrameItem is one ranked entry in a replay frame.
type FrameItem struct {
	Item   string `json:"item"`
	Approx int64  `json:"approx"`
	Exact  int64  `json:"exact"`
}

// Frame is one replay snapshot of the sliding-window state.
type Frame struct {
	Now       int64       `json:"now"`
	WinStart  int64       `json:"ws"`
	Total     int64       `json:"total"`
	Distinct  int         `json:"distinct"`
	Top       []FrameItem `json:"top"`  // approximate Top-K (sketch estimates)
	ExactTop  []FrameItem `json:"etop"` // ground-truth Top-K (exact counts)
	MaxErr    int64       `json:"maxErr"`
	Bound     int64       `json:"bound"`
	LiveTotal int64       `json:"live"`
	Match     bool        `json:"match"` // approx Top-K set == exact Top-K set
}

// ReportMeta describes the run configuration embedded into the report.
type ReportMeta struct {
	Window     int64   `json:"window"`
	K          int     `json:"k"`
	Subwindows int     `json:"subwindows"`
	Epsilon    float64 `json:"epsilon"`
	Delta      float64 `json:"delta"`
	Events     int     `json:"events"`
	Counters   int     `json:"counters"`
}

// BuildFrame snapshots the tracker. Verification work (exact Top-K, max
// error over all distinct items) is done here, offline — never on the
// tracker's hot path.
func BuildFrame(t *Tracker, now int64, k int) Frame {
	exact := t.ExactCounts()
	etop := ExactTopK(exact, k)

	f := Frame{
		Now:       now,
		WinStart:  now - t.Window().Size(),
		Total:     t.WindowTotal(),
		Distinct:  t.Distinct(),
		Bound:     t.ErrorBound(),
		LiveTotal: t.LiveTotal(),
	}
	inApprox := make(map[string]bool, k)
	for _, ic := range t.TopK() {
		inApprox[ic.Item] = true
		f.Top = append(f.Top, FrameItem{Item: ic.Item, Approx: ic.Count, Exact: exact[ic.Item]})
	}
	inExact := make(map[string]bool, k)
	for _, ic := range etop {
		inExact[ic.Item] = true
		f.ExactTop = append(f.ExactTop, FrameItem{Item: ic.Item, Approx: t.Estimate(ic.Item), Exact: ic.Count})
	}
	f.Match = len(inApprox) == len(inExact)
	for it := range inApprox {
		if !inExact[it] {
			f.Match = false
			break
		}
	}
	var maxErr int64
	for it, c := range exact {
		if d := t.Estimate(it) - c; d > maxErr {
			maxErr = d
		}
	}
	f.MaxErr = maxErr
	return f
}

// RenderHTML writes a single self-contained HTML report (no external
// resources, no frameworks) that replays the frames and plots the error
// curve against its theoretical bound.
func RenderHTML(w io.Writer, meta ReportMeta, frames []Frame) error {
	payload, err := json.Marshal(struct {
		Meta   ReportMeta `json:"meta"`
		Frames []Frame    `json:"frames"`
	}{Meta: meta, Frames: frames})
	if err != nil {
		return err
	}
	page := strings.Replace(reportTemplate, "/*__DATA__*/", string(payload), 1)
	if strings.Contains(page, "/*__DATA__*/") {
		return fmt.Errorf("topk: data placeholder not substituted")
	}
	_, err = io.WriteString(w, page)
	return err
}

const reportTemplate = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>滑动时间窗口 Top-K 回放报告</title>
<style>
  :root {
    color-scheme: light;
    --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
    --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
    --approx:#2a78d6; --exact:#eb6834; --good:#006300; --bad:#b00020; --bound:#b00020;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--page); color:var(--ink);
         font:14px/1.5 -apple-system, "Helvetica Neue", "PingFang SC", "Microsoft YaHei", sans-serif; }
  header { padding:20px 24px 8px; }
  h1 { font-size:20px; margin:0 0 4px; }
  h2 { font-size:13px; margin:0 0 10px; color:var(--ink-2); font-weight:600;
       text-transform:uppercase; letter-spacing:.04em; }
  .meta { color:var(--muted); font-size:12.5px; }
  .controls { display:flex; gap:12px; align-items:center; padding:10px 24px;
              position:sticky; top:0; background:var(--page); z-index:2;
              border-bottom:1px solid var(--border); }
  button { font:inherit; padding:5px 14px; border:1px solid var(--border);
           background:var(--surface); border-radius:6px; cursor:pointer; }
  button:hover { border-color:var(--axis); }
  select { font:inherit; padding:4px 6px; border:1px solid var(--border);
           border-radius:6px; background:var(--surface); }
  input[type=range] { flex:1; accent-color:var(--approx); }
  .framelabel { font-variant-numeric:tabular-nums; color:var(--ink-2);
                font-size:12.5px; white-space:nowrap; }
  main { display:grid; grid-template-columns:minmax(0,3fr) minmax(0,2fr);
         gap:16px; padding:16px 24px; }
  @media (max-width: 860px) { main { grid-template-columns:1fr; } }
  .panel { background:var(--surface); border:1px solid var(--border);
           border-radius:10px; padding:14px 16px; }
  .row { display:grid; grid-template-columns:28px minmax(90px,150px) 1fr auto;
         gap:8px; align-items:center; padding:3px 0; }
  .rank { color:var(--muted); font-variant-numeric:tabular-nums; text-align:right; }
  .item { font-family:ui-monospace, Menlo, monospace; font-size:12.5px;
          overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .bars { position:relative; height:16px; }
  .bar { position:absolute; left:0; top:0; height:16px; border-radius:3px; }
  .bar.approx { background:var(--approx); opacity:.85; }
  .bar.exact  { background:transparent; border:2px solid var(--exact);
                height:12px; top:0; }
  .nums { font-variant-numeric:tabular-nums; font-size:12px; color:var(--ink-2);
          white-space:nowrap; }
  .nums .err { color:var(--bad); }
  .tag { display:inline-block; font-size:10.5px; padding:0 5px; border-radius:4px;
         margin-left:6px; vertical-align:1px; }
  .tag.only-approx { background:rgba(42,120,214,.12); color:var(--approx); }
  .tag.only-exact  { background:rgba(235,104,52,.12); color:var(--exact); }
  .legend { font-size:12px; color:var(--ink-2); margin-bottom:8px; }
  .sw { display:inline-block; width:12px; height:12px; border-radius:3px;
        vertical-align:-2px; margin:0 4px 0 12px; }
  .sw.approx { background:var(--approx); }
  .sw.exact  { border:2px solid var(--exact); }
  .stats { display:grid; grid-template-columns:auto 1fr; gap:4px 14px; font-size:13px; }
  .stats dt { color:var(--muted); } .stats dd { margin:0; font-variant-numeric:tabular-nums; }
  .ok { color:var(--good); font-weight:600; }
  .bad { color:var(--bad); font-weight:600; }
  .bottom { padding:0 24px 28px; }
  canvas { width:100%; display:block; }
  .kbd { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>滑动时间窗口 Top-K 回放报告</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="controls">
  <button id="play">▶ 播放</button>
  <select id="speed">
    <option value="1">1×</option><option value="2">2×</option>
    <option value="4" selected>4×</option><option value="8">8×</option>
    <option value="16">16×</option>
  </select>
  <input type="range" id="scrub" min="0" value="0">
  <span class="framelabel" id="flabel"></span>
  <span class="kbd">空格:播放/暂停 ←/→:步进</span>
</div>
<main>
  <div class="panel">
    <h2>Top-K 排名（近似 vs 精确）</h2>
    <div class="legend">图例:<span class="sw approx"></span>近似计数(滑窗 CMS)<span class="sw exact"></span>精确计数(增量滑窗)</div>
    <div id="bars"></div>
  </div>
  <div class="panel">
    <h2>窗口状态</h2>
    <dl class="stats" id="stats"></dl>
  </div>
</main>
<div class="bottom">
  <div class="panel">
    <h2>近似计数误差曲线（全部活跃项的最大误差 vs 理论上界 ε·N_live）</h2>
    <canvas id="errc" height="220"></canvas>
  </div>
</div>
<script>
var DATA = /*__DATA__*/;
var frames = DATA.frames, meta = DATA.meta;
var idx = 0, playing = false, timer = null;

document.getElementById('meta').textContent =
  '窗口 ' + meta.window + ' 时间单位 · K=' + meta.k + ' · 子窗口 ' + meta.subwindows +
  ' · ε=' + meta.epsilon + ' · δ=' + meta.delta + ' · 事件 ' + meta.events +
  ' · Sketch 计数器 ' + meta.counters + ' 个(常量,与事件总数无关)';

var scrub = document.getElementById('scrub');
scrub.max = frames.length - 1;

function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;'); }

function render(i) {
  idx = Math.max(0, Math.min(frames.length - 1, i));
  var f = frames[idx];
  scrub.value = idx;
  document.getElementById('flabel').textContent =
    '帧 ' + (idx+1) + '/' + frames.length + ' · t=' + f.now;

  // ---- Top-K bars: union of approx and exact top-k, approx rank order ----
  var exactSet = {}, approxSet = {}, j, it;
  for (j = 0; j < f.etop.length; j++) exactSet[f.etop[j].item] = true;
  for (j = 0; j < f.top.length; j++) approxSet[f.top[j].item] = true;
  var rows = [];
  for (j = 0; j < f.top.length; j++) {
    it = f.top[j];
    rows.push({item: it.item, approx: it.approx, exact: it.exact,
               inA: true, inE: !!exactSet[it.item]});
  }
  for (j = 0; j < f.etop.length; j++) {
    it = f.etop[j];
    if (!approxSet[it.item])
      rows.push({item: it.item, approx: it.approx, exact: it.exact,
                 inA: false, inE: true});
  }
  var maxc = 1;
  for (j = 0; j < rows.length; j++) maxc = Math.max(maxc, rows[j].approx, rows[j].exact);
  var html = '';
  for (j = 0; j < rows.length; j++) {
    var row = rows[j];
    var wa = (100 * row.approx / maxc).toFixed(2);
    var we = (100 * row.exact / maxc).toFixed(2);
    var err = row.approx - row.exact;
    var tag = row.inA && !row.inE ? '<span class="tag only-approx">仅近似榜</span>'
            : !row.inA && row.inE ? '<span class="tag only-exact">仅精确榜</span>' : '';
    html += '<div class="row"><span class="rank">' + (j+1) + '</span>' +
      '<span class="item" title="' + esc(row.item) + '">' + esc(row.item) + tag + '</span>' +
      '<span class="bars"><span class="bar approx" style="width:' + wa + '%"></span>' +
      '<span class="bar exact" style="width:' + we + '%"></span></span>' +
      '<span class="nums">' + row.approx + ' / ' + row.exact +
      ' <span class="err">+' + err + '</span></span></div>';
  }
  document.getElementById('bars').innerHTML = html;

  // ---- stats ----
  var st = document.getElementById('stats');
  st.innerHTML =
    '<dt>当前时间</dt><dd>' + f.now + '</dd>' +
    '<dt>窗口区间</dt><dd>(' + f.ws + ', ' + f.now + ']</dd>' +
    '<dt>窗口内事件</dt><dd>' + f.total + '</dd>' +
    '<dt>活跃项数</dt><dd>' + f.distinct + '</dd>' +
    '<dt>N_live(桶内总计)</dt><dd>' + f.live + '</dd>' +
    '<dt>最大误差</dt><dd>' + f.maxErr + '</dd>' +
    '<dt>误差上界 ε·N_live</dt><dd>' + f.bound + '</dd>' +
    '<dt>上界守住</dt><dd class="' + (f.maxErr <= f.bound ? 'ok' : 'bad') + '">' +
      (f.maxErr <= f.bound ? '是' : '否') + '</dd>' +
    '<dt>Top-K 集合一致</dt><dd class="' + (f.match ? 'ok' : 'bad') + '">' +
      (f.match ? '一致' : '有出入(近似误差所致)') + '</dd>';

  drawErr();
}

// ---- error curve canvas ----
var cvs = document.getElementById('errc');
function drawErr() {
  var dpr = window.devicePixelRatio || 1;
  var W = cvs.clientWidth, H = 220;
  if (cvs.width !== W * dpr) { cvs.width = W * dpr; cvs.height = H * dpr; }
  var ctx = cvs.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  var padL = 56, padR = 12, padT = 12, padB = 26;
  var iw = W - padL - padR, ih = H - padT - padB;
  var ymax = 1, i;
  for (i = 0; i < frames.length; i++) ymax = Math.max(ymax, frames[i].bound, frames[i].maxErr);
  ymax *= 1.08;
  function X(i){ return padL + iw * i / Math.max(1, frames.length - 1); }
  function Y(v){ return padT + ih * (1 - v / ymax); }
  // grid + y labels
  ctx.strokeStyle = '#e1e0d9'; ctx.fillStyle = '#898781';
  ctx.font = '11px ui-monospace, Menlo, monospace'; ctx.textAlign = 'right';
  ctx.lineWidth = 1;
  for (var g = 0; g <= 4; g++) {
    var v = ymax * g / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(String(Math.round(v)), padL - 6, y + 4);
  }
  // bound line (red dashed)
  ctx.strokeStyle = '#b00020'; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (i = 0; i < frames.length; i++) { var x = X(i), yb = Y(frames[i].bound);
    if (i === 0) ctx.moveTo(x, yb); else ctx.lineTo(x, yb); }
  ctx.stroke();
  // max-err curve (blue)
  ctx.strokeStyle = '#2a78d6'; ctx.setLineDash([]); ctx.lineWidth = 1.8;
  ctx.beginPath();
  for (i = 0; i < frames.length; i++) { var x2 = X(i), ye = Y(frames[i].maxErr);
    if (i === 0) ctx.moveTo(x2, ye); else ctx.lineTo(x2, ye); }
  ctx.stroke();
  // cursor
  ctx.strokeStyle = '#0b0b0b'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(idx), padT); ctx.lineTo(X(idx), padT + ih); ctx.stroke();
  // legend
  ctx.textAlign = 'left'; ctx.fillStyle = '#2a78d6';
  ctx.fillText('— 最大误差', padL + 8, padT + 12);
  ctx.fillStyle = '#b00020';
  ctx.fillText('┄ 上界 ε·N_live', padL + 92, padT + 12);
  ctx.fillStyle = '#898781';
  ctx.fillText('t=' + frames[0].now, padL, H - 8);
  var end = 't=' + frames[frames.length-1].now;
  ctx.fillText(end, W - padR - ctx.measureText(end).width, H - 8);
}

// ---- playback ----
var playBtn = document.getElementById('play');
var speedSel = document.getElementById('speed');
function step() {
  var n = parseInt(speedSel.value, 10);
  if (idx >= frames.length - 1) { pause(); return; }
  render(idx + n);
}
function play() { playing = true; playBtn.textContent = '⏸ 暂停';
  timer = setInterval(step, 90); }
function pause() { playing = false; playBtn.textContent = '▶ 播放';
  clearInterval(timer); }
playBtn.onclick = function(){ playing ? pause() : play(); };
scrub.oninput = function(){ pause(); render(parseInt(scrub.value, 10)); };
document.addEventListener('keydown', function(e){
  if (e.code === 'Space') { e.preventDefault(); playing ? pause() : play(); }
  else if (e.key === 'ArrowRight') { pause(); render(idx + 1); }
  else if (e.key === 'ArrowLeft') { pause(); render(idx - 1); }
});
window.addEventListener('resize', drawErr);

render(0);
</script>
</body>
</html>
`
