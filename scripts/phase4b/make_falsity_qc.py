#!/usr/bin/env python3
"""Sample 100 clean negatives and write falsity QC TSV + HTML."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QC = ROOT / "data/phase4a/qc"
SEED = 42
DEFAULT_ROOT = "D:/works/relation_hallu/data/relsim_images"

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 4B Clean Negative 视觉假性 QC（100）</title>
<style>
  :root { --bg:#0f1419; --card:#1a2330; --ink:#e8eef6; --muted:#9aa8b7; --yes:#2f9e73; --no:#c45c5c; --acc:#4c8dff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif; background:var(--bg); color:var(--ink); }
  header { position:sticky; top:0; z-index:5; background:#121a24; border-bottom:1px solid #2a3646; padding:12px 20px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  .progress { color:var(--muted); font-size:13px; }
  button { background:#243044; color:var(--ink); border:1px solid #3a4a5e; border-radius:8px; padding:8px 12px; cursor:pointer; }
  button.primary { background:var(--acc); border-color:var(--acc); color:#fff; }
  main { max-width:1100px; margin:0 auto; padding:20px; }
  .card { background:var(--card); border-radius:14px; padding:16px; display:grid; grid-template-columns: minmax(280px, 1fr) 1.1fr; gap:18px; }
  img { width:100%; max-height:520px; object-fit:contain; background:#0b0f14; border-radius:10px; }
  .meta { color:var(--muted); font-size:12px; margin:8px 0 0; word-break:break-all; }
  .stmt { padding:10px 12px; border-radius:10px; margin:0 0 10px; }
  .pos { background:#173528; }
  .neg { background:#3a1f24; }
  .yn { display:flex; gap:8px; flex-wrap:wrap; }
  .yn label { display:flex; align-items:center; gap:6px; padding:6px 10px; border-radius:8px; border:1px solid #3a4a5e; cursor:pointer; }
  .yn label.on-no { border-color:var(--yes); background:#173528; }
  .yn label.on-yes { border-color:var(--no); background:#3a1f24; }
  .yn label.on-unk { border-color:#c9a227; background:#3a3420; }
  .jump { width:64px; background:#121a24; color:var(--ink); border:1px solid #3a4a5e; border-radius:8px; padding:7px 8px; }
  .imgroot { flex:1; min-width:280px; background:#121a24; color:var(--ink); border:1px solid #3a4a5e; border-radius:8px; padding:7px 10px; }
  .hintbar { width:100%; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>Phase 4B 负例视觉假性</h1>
  <div class="progress" id="progress"></div>
  <button id="prev">上一张</button>
  <input class="jump" id="jump" type="number" min="1" max="100">
  <button id="next">下一张</button>
  <button class="primary" id="save">导出 TSV</button>
  <input class="imgroot" id="imgroot" spellcheck="false">
  <div class="hintbar">只判断 Negative 关系在图中是否成立。默认图片目录 D:\works\relation_hallu\data\relsim_images。Chrome 若无图，用 Edge/Firefox，或把 html 拷进图片目录并把路径改成 .</div>
</header>
<main><div class="card" id="card"></div></main>
<script>
const ITEMS = __DATA__;
const KEY = "phase4b_falsity_v1";
const ROOT_KEY = "phase4a_qc_imgroot";
const DEFAULT_ROOT = "D:/works/relation_hallu/data/relsim_images";
const store = JSON.parse(localStorage.getItem(KEY) || "{}");
let idx = 0;
function imageSrc(name) {
  let root = (document.getElementById("imgroot").value || DEFAULT_ROOT).trim().replace(/\\/g, "/");
  if (!root || root === ".") return name;
  if (!root.endsWith("/")) root += "/";
  if (/^[A-Za-z]:\//.test(root)) return "file:///" + root + name;
  return root + name;
}
function recId() { return ITEMS[idx].id; }
function val() { return (store[recId()] || {}).human_falsity || ""; }
function setVal(v) {
  store[recId()] = store[recId()] || {};
  store[recId()].human_falsity = v;
  localStorage.setItem(KEY, JSON.stringify(store));
  render();
}
function filledCount() { return ITEMS.filter(it => (store[it.id] || {}).human_falsity).length; }
function render() {
  const it = ITEMS[idx];
  document.getElementById("jump").value = idx + 1;
  document.getElementById("progress").textContent = `${idx+1} / ${ITEMS.length}　family=${it.family}　已填 ${filledCount()}/100`;
  const v = val();
  document.getElementById("card").innerHTML = `
    <div>
      <img src="${imageSrc(it.image_file)}" alt="${it.id}" onerror="this.nextElementSibling.style.display='block'">
      <p class="meta" style="display:none;color:#c45c5c">图片未加载：${imageSrc(it.image_file)}</p>
      <p class="meta">${it.id}<br>${it.image_file}</p>
    </div>
    <div>
      <div class="stmt pos"><b>正例（应成立）　${it.positive_relation}</b>${it.positive_statement}</div>
      <div class="stmt neg"><b>负例（声称不成立）　${it.negative_relation}</b>${it.negative_statement}</div>
      <p class="meta">InternVL 二次 pass：${it.internvl_judgment || "尚未跑"}</p>
      <div class="yn">
        <label class="${v==="不成立"?"on-no":""}"><input type="radio" name="h" value="不成立" ${v==="不成立"?"checked":""}> 不成立（标签正确）</label>
        <label class="${v==="成立"?"on-yes":""}"><input type="radio" name="h" value="成立" ${v==="成立"?"checked":""}> 成立（标签噪声）</label>
        <label class="${v==="无法判断"?"on-unk":""}"><input type="radio" name="h" value="无法判断" ${v==="无法判断"?"checked":""}> 无法判断</label>
      </div>
    </div>`;
  document.querySelectorAll(".yn input").forEach(el => el.addEventListener("change", () => setVal(el.value)));
}
function exportTsv() {
  const fields = ["id","image_path","family","subject","object","positive_relation","positive_statement","negative_relation","negative_statement","internvl_judgment","internvl_raw","human_falsity"];
  const lines = [fields.join("\t")];
  for (const it of ITEMS) {
    const a = store[it.id] || {};
    const row = {...it, human_falsity: a.human_falsity || it.human_falsity || ""};
    lines.push(fields.map(k => (row[k] || "").toString().replace(/\t|\n/g, " ")).join("\t"));
  }
  const blob = new Blob([lines.join("\n")+"\n"], {type: "text/tab-separated-values"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = "review_100_falsity_filled.tsv"; a.click();
}
document.getElementById("prev").onclick = () => { idx = (idx + ITEMS.length - 1) % ITEMS.length; render(); };
document.getElementById("next").onclick = () => { idx = (idx + 1) % ITEMS.length; render(); };
document.getElementById("jump").onchange = (e) => {
  idx = Math.min(ITEMS.length, Math.max(1, parseInt(e.target.value || "1", 10))) - 1; render();
};
document.getElementById("imgroot").value = localStorage.getItem(ROOT_KEY) || DEFAULT_ROOT;
document.getElementById("imgroot").addEventListener("change", () => {
  localStorage.setItem(ROOT_KEY, document.getElementById("imgroot").value); render();
});
document.getElementById("save").onclick = exportTsv;
document.addEventListener("keydown", (e) => {
  if (["INPUT","TEXTAREA"].includes(e.target.tagName)) return;
  if (e.key === "ArrowLeft") document.getElementById("prev").click();
  if (e.key === "ArrowRight") document.getElementById("next").click();
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    tsv_in = QC / "review_300.tsv"
    with tsv_in.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    buckets = defaultdict(list)
    for rec in rows:
        buckets[rec.get("family") or "other"].append(rec)
    rng = random.Random(SEED)
    target = {"spatial": 34, "contact": 33, "other": 33}
    picked = []
    for fam, k in target.items():
        pool = list(buckets.get(fam) or [])
        rng.shuffle(pool)
        if len(pool) < k:
            raise SystemExit(f"family {fam} only has {len(pool)} < {k}")
        picked.extend(pool[:k])
    payload = []
    for rec in picked:
        name = Path(rec["image_path"]).name
        payload.append(
            {
                "id": rec["id"],
                "image_path": rec["image_path"],
                "image_file": name,
                "family": rec.get("family") or "",
                "subject": rec.get("subject") or "",
                "object": rec.get("object") or "",
                "positive_relation": rec.get("positive_relation") or "",
                "positive_statement": rec.get("positive_statement") or "",
                "negative_relation": rec.get("negative_relation") or "",
                "negative_statement": rec.get("negative_statement") or "",
                "internvl_judgment": "",
                "internvl_raw": "",
                "human_falsity": "",
            }
        )
    tsv_out = QC / "review_100_falsity.tsv"
    fields = [
        "id",
        "image_path",
        "family",
        "subject",
        "object",
        "positive_relation",
        "positive_statement",
        "negative_relation",
        "negative_statement",
        "internvl_judgment",
        "internvl_raw",
        "human_falsity",
    ]
    with tsv_out.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for rec in payload:
            f.write("\t".join((rec.get(k) or "").replace("\t", " ") for k in fields) + "\n")
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    (QC / "review_100_falsity.html").write_text(html, encoding="utf-8")
    print(f"wrote {tsv_out} n={len(payload)}", flush=True)
    print(f"wrote {QC / 'review_100_falsity.html'}", flush=True)
    fam_c = Counter(r["family"] for r in payload)
    print("families", dict(fam_c), flush=True)


if __name__ == "__main__":
    main()
