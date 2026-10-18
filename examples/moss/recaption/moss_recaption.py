"""MOSS-Image style recaptioning for a LoRA dataset (OpenMOSS USERGUIDE "Image Captioning", 2026-10-16).

The official recipe: a VLM rewrites every caption as JSON with long / medium / short descriptions (prompt_official.txt),
and training mixes the original caption (20%) with the recaptions (80%, the three lengths 1:1:1).

Steps. Every step can be re-run; nothing in a dataset is written without --apply, and the source dataset never is.
  copy   copy the subsets of a dataset config into a work folder (images + captions; originals also in _orig/) and write
         a dataset config for the copy with enable_wildcard = true
  queue  list the copy's images with their reference caption (queue.jsonl); --pilot N also picks N of them (pilot.jsonl)
  run    caption the queued images with Codex CLI, one call per image (prompt + image, JSON schema), into
         <image stem>.moss_<prompt>.json beside each image; resumable (progress_run_<prompt>.jsonl)
  check  validate the JSON files and write review_<prompt>.md
  build  write the training captions in the copy: the original caption, then "<trigger> ||| ||| <caption>" lines for
         short / medium / long (default 1:2:2:2 lines; see README.md for the resulting mix)

Run with the sd_scripts venv Python (3.11+). Paths may contain non-ASCII characters; every file is UTF-8 without BOM.
"""

import argparse
import datetime
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import toml as tomllib  # type: ignore

HERE = Path(__file__).resolve().parent
PROMPTS = {"official": HERE / "prompt_official.txt", "no_style": HERE / "prompt_no_style.txt"}
SCHEMA = HERE / "caption_schema.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
ORIG = "_orig"
FIELDS = ("short", "medium", "long")
BORDERS = {"none", "left", "right", "up", "down"}
STYLE_WORDS = re.compile(
    r"\b(illustrat\w*|paint\w*|anime|manga|art ?style|style[ds]?|stylized|drawn|drawing|digital art|artwork|cel[- ]shad\w*|"
    r"shading|line ?art|linework|lineart|brush\w*|watercolou?r|render\w*|sketch\w*|ink(ed|ing)?|airbrush\w*|"
    r"semi-realistic|realistic|painterly|soft[- ]focus)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- helpers


def log(message: str) -> None:
    print(message, flush=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> list:
    if not path.is_file():
        return []
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def images_in(folder: Path) -> list:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a TOML basic string
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {value!r} to TOML")


def is_inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def work_subsets(work: Path) -> list:
    manifest = work / "manifest.json"
    if not manifest.is_file():
        sys.exit(f"no {manifest}: run 'copy --apply' first")
    return json.loads(read_text(manifest))["subsets"]


def reference_text(caption: str, trigger: str) -> str:
    """The original caption for the VLM: tags and description, without the trigger and the ||| separators."""
    caption = " ".join(caption.split("\n")).replace("\\(", "(").replace("\\)", ")")
    parts = [p.strip() for p in caption.split("|||")]
    tag_parts, description = (parts[:-1], parts[-1]) if len(parts) >= 3 else (parts, "")

    def keep(token: str) -> bool:
        return token and token.lower() != trigger.lower() and not token.startswith("@")

    tags = [t.strip() for part in tag_parts for t in part.split(",") if keep(t.strip())]
    if trigger:
        description = re.sub(re.escape(trigger) + r",?\s*", "", description, flags=re.IGNORECASE).strip()
    lines = []
    if tags:
        lines.append("Tags: " + ", ".join(tags))
    if description:
        lines.append("Description: " + description)
    return "\n".join(lines) or "(none)"


def validate(data) -> list:
    problems = []
    if not isinstance(data, dict):
        return ["not a JSON object"]
    caption = data.get("caption")
    if not isinstance(caption, dict):
        problems.append("caption missing")
    else:
        for field in FIELDS:
            if not isinstance(caption.get(field), str) or not caption[field].strip():
                problems.append(f"caption.{field} empty")
    marks = data.get("watermark")
    if not isinstance(marks, list):
        problems.append("watermark is not a list")
    else:
        for mark in marks:
            box = mark.get("box_2d") if isinstance(mark, dict) else None
            if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) and 0 <= v <= 1000 for v in box)
                    and box[0] <= box[2] and box[1] <= box[3]):
                problems.append(f"watermark box {box!r}")
    borders = data.get("borders")
    if not (isinstance(borders, list) and borders and set(borders) <= BORDERS and ("none" not in borders or borders == ["none"])):
        problems.append(f"borders {borders!r}")
    return problems


def find_codex(explicit) -> str:
    if explicit:
        return explicit
    root = Path(os.environ.get("LOCALAPPDATA", "")) / "OpenAI" / "Codex" / "bin"
    found = sorted(root.rglob("codex.exe"), key=lambda p: p.stat().st_mtime, reverse=True) if root.is_dir() else []
    if found:
        return str(found[0])
    path = shutil.which("codex")
    if path:
        return path
    sys.exit("codex not found: pass --codex <path to codex.exe>")


def check_chatgpt_login() -> None:
    """Captioning runs on the ChatGPT plan. Refuse API-key auth so a run never bills the OpenAI API."""
    auth = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json"
    try:
        mode = json.loads(read_text(auth)).get("auth_mode")
    except (OSError, ValueError):
        mode = None
    if mode != "chatgpt":
        sys.exit(f"Codex is not signed in with ChatGPT ({auth}: auth_mode={mode!r}); refusing to use API-key billing")


# ---------------------------------------------------------------- copy


def cmd_copy(args) -> None:
    config_path = Path(args.dataset_config)
    config = tomllib.loads(read_text(config_path))
    general = config.get("general", {})
    work = Path(args.work).resolve()
    plan, names = [], {}
    for dataset in config.get("datasets", []):
        for subset in dataset.get("subsets", []):
            source = Path(subset["image_dir"])
            if not source.is_dir():
                sys.exit(f"not a folder: {source}")
            if is_inside(work, source) or is_inside(source, work):
                sys.exit(f"the work folder must be outside the dataset: {work} / {source}")
            name, key = source.name, os.path.normcase(str(source.resolve()))
            ext = subset.get("caption_extension", dataset.get("caption_extension", general.get("caption_extension", ".caption")))
            if name.lower() in names:
                if names[name.lower()] != (key, ext):  # the same folder in several datasets (resolutions) is copied once
                    sys.exit(f"two different subset folders (or caption extensions) share the name {name!r}")
                continue
            if name == ORIG:
                sys.exit(f"a subset folder is named {ORIG!r}")
            names[name.lower()] = (key, ext)
            files = []
            for image in images_in(source):
                caption = image.with_suffix(ext)
                files.append((image, caption if caption.is_file() else None))
            plan.append({"name": name, "source": source, "ext": ext, "files": files})

    total = sum(len(s["files"]) for s in plan)
    size = sum(i.stat().st_size + (c.stat().st_size if c else 0) for s in plan for i, c in s["files"])
    log(f"source config: {config_path.resolve()}")
    log(f"work folder:   {work}")
    for s in plan:
        missing = sum(1 for _, c in s["files"] if c is None)
        log(f"  {s['name']}: {len(s['files'])} images ({missing} without caption) <- {s['source']}")
    log(f"total {total} images, {size / 2**20:.1f} MiB; originals of the captions also go to {work / ORIG}")
    if not args.apply:
        log("dry run: nothing copied (add --apply)")
        return

    copied = skipped = 0
    for s in plan:
        for image, caption in s["files"]:
            targets = [(image, work / s["name"] / image.name)]
            if caption:
                targets += [(caption, work / s["name"] / caption.name), (caption, work / ORIG / s["name"] / caption.name)]
            for source, target in targets:
                if target.exists():
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied += 1

    manifest = {
        "source_config": str(config_path.resolve()),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "subsets": [{"name": s["name"], "source": str(s["source"]), "caption_extension": s["ext"],
                     "images": [i.name for i, _ in s["files"]]} for s in plan],
    }
    write_text(work / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")

    out = [f"# MOSS recaption copy of {config_path.resolve()} ({manifest['created']}).",
           "# enable_wildcard: every caption line is one alternative (line 1 is the original caption).", "[general]"]
    general = dict(general, enable_wildcard=True)
    out += [f"{k} = {toml_value(v)}" for k, v in general.items()]
    for dataset in config.get("datasets", []):
        out += ["", "[[datasets]]"] + [f"{k} = {toml_value(v)}" for k, v in dataset.items() if k != "subsets"]
        for subset in dataset.get("subsets", []):
            subset = dict(subset, image_dir=(work / Path(subset["image_dir"]).name).as_posix())
            out += ["", "  [[datasets.subsets]]"] + [f"  {k} = {toml_value(v)}" for k, v in subset.items()]
    config_out = work / "dataset_config_moss_recap.toml"
    write_text(config_out, "\n".join(out) + "\n")
    log(f"copied {copied} files ({skipped} already there); manifest {work / 'manifest.json'}; dataset config {config_out}")


# ---------------------------------------------------------------- queue


def cmd_queue(args) -> None:
    work = Path(args.work).resolve()
    items = []
    for subset in work_subsets(work):
        for name in subset["images"]:
            image = work / subset["name"] / name
            original = work / ORIG / subset["name"] / (Path(name).stem + subset["caption_extension"])
            caption = read_text(original).strip() if original.is_file() else ""
            items.append({"id": f"{subset['name']}/{Path(name).stem}", "subset": subset["name"], "image": str(image),
                          "reference": reference_text(caption, args.trigger)})
    write_text(work / "queue.jsonl", "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    log(f"{len(items)} images -> {work / 'queue.jsonl'}")
    if not args.pilot:
        return
    pilot_path = work / "pilot.jsonl"
    if pilot_path.is_file() and not args.repick:
        log(f"pilot kept: {pilot_path} (--repick to choose again)")
        return
    rng = random.Random(args.seed)
    by_subset = {}
    for item in items:
        by_subset.setdefault(item["subset"], []).append(item)
    share = {k: len(v) * args.pilot / len(items) for k, v in by_subset.items()}
    count = {k: int(v) for k, v in share.items()}
    for k in sorted(share, key=lambda k: share[k] - count[k], reverse=True)[: args.pilot - sum(count.values())]:
        count[k] += 1  # largest remainders
    pilot = [item for k, v in by_subset.items() for item in rng.sample(v, min(count[k], len(v)))]
    write_text(pilot_path, "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in pilot))
    log(f"pilot {len(pilot)} ({', '.join(f'{k} {count[k]}' for k in by_subset)}) -> {pilot_path}")


# ---------------------------------------------------------------- run


def output_path(image: Path, prompt: str) -> Path:
    return image.with_name(f"{image.stem}.moss_{prompt}.json")


def parse_reply(text: str):
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)


def cmd_run(args) -> None:
    work = Path(args.work).resolve()
    queue = Path(args.queue)
    queue = queue if queue.is_absolute() else work / queue
    items = read_jsonl(queue)
    template = read_text(PROMPTS[args.prompt])
    codex = find_codex(args.codex)
    check_chatgpt_login()
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY")}
    progress = work / f"progress_run_{args.prompt}.jsonl"
    todo = [i for i in items if args.redo or not output_path(Path(i["image"]), args.prompt).is_file()]
    if args.limit:
        todo = todo[: args.limit]
    log(f"{len(items)} queued, {len(items) - len(todo)} done before; captioning {len(todo)} with {args.model} "
        f"(effort {args.effort}, prompt {args.prompt}); progress {progress}")
    ok = failed = 0
    with tempfile.TemporaryDirectory(prefix="moss_recaption_") as scratch:
        for n, item in enumerate(todo, 1):
            image = Path(item["image"])
            reply = Path(scratch) / f"reply_{n}.txt"
            command = [codex, "exec", "-m", args.model, "-c", f'model_reasoning_effort="{args.effort}"',
                       "-i", str(image), "--output-schema", str(SCHEMA), "-o", str(reply), "-s", args.sandbox,
                       "--skip-git-repo-check", "--ephemeral", "-C", scratch, "-"]
            prompt = template.replace("{{reference}}", item["reference"])
            started = datetime.datetime.now()
            record = {"id": item["id"], "prompt": args.prompt, "model": args.model, "effort": args.effort,
                      "started": started.isoformat(timespec="seconds")}
            try:
                result = subprocess.run(command, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                        errors="replace", env=env, timeout=args.timeout)
                data = parse_reply(read_text(reply)) if reply.is_file() else None
                problems = validate(data) if data is not None else [f"no reply (exit {result.returncode})"]
                if data is not None and not problems:
                    write_text(output_path(image, args.prompt), json.dumps(data, ensure_ascii=False, indent=2) + "\n")
                    record["status"] = "ok"
                    ok += 1
                else:
                    if data is not None:
                        write_text(output_path(image, args.prompt).with_suffix(".invalid.json"),
                                   json.dumps(data, ensure_ascii=False, indent=2) + "\n")
                    record.update(status="error", problems=problems, stderr=result.stderr[-1500:])
                    failed += 1
            except (subprocess.TimeoutExpired, ValueError, OSError) as error:
                record.update(status="error", problems=[f"{type(error).__name__}: {error}"[:500]])
                failed += 1
            record["seconds"] = round((datetime.datetime.now() - started).total_seconds(), 1)
            append_jsonl(progress, record)
            log(f"[{n}/{len(todo)}] {item['id']}: {record['status']} ({record['seconds']} s)"
                + ("" if record["status"] == "ok" else f" {record['problems']}"))
    log(f"done: {ok} ok, {failed} failed; outputs beside the images as *.moss_{args.prompt}.json")


# ---------------------------------------------------------------- check


def cmd_check(args) -> None:
    work = Path(args.work).resolve()
    trigger = args.trigger.lower()
    rows, totals = [], {"files": 0, "problems": 0, "style": 0, "watermarks": 0, "borders": 0}
    words = {field: [] for field in FIELDS}
    for subset in work_subsets(work):
        for name in subset["images"]:
            image = work / subset["name"] / name
            path = output_path(image, args.prompt)
            if not path.is_file():
                continue
            totals["files"] += 1
            data = json.loads(read_text(path))
            problems = validate(data)
            caption = data.get("caption", {}) if isinstance(data, dict) else {}
            text = " ".join(str(caption.get(f, "")) for f in FIELDS)
            if trigger and trigger in text.lower():
                problems.append("trigger in caption")
            if "|||" in text:
                problems.append("'|||' in caption")
            if "watermark" in text.lower():
                problems.append("mentions a watermark")
            style = sorted({m.group(0).lower() for m in STYLE_WORDS.finditer(text)})
            for field in FIELDS:
                words[field].append(len(str(caption.get(field, "")).split()))
            totals["problems"] += bool(problems)
            totals["style"] += bool(style)
            totals["watermarks"] += bool(data.get("watermark"))
            totals["borders"] += data.get("borders") not in (["none"], None)
            rows.append((f"{subset['name']}/{Path(name).stem}", caption.get("short", ""),
                         "/".join(str(words[f][-1]) for f in ("long", "medium", "short")), ", ".join(style),
                         len(data.get("watermark") or []), ",".join(data.get("borders") or []), "; ".join(problems)))
    mean = {f: (sum(v) / len(v) if v else 0) for f, v in words.items()}
    report = [f"# MOSS recaption review: prompt {args.prompt}", "",
              f"{totals['files']} files, {totals['problems']} with problems, {totals['style']} use style words, "
              f"{totals['watermarks']} with watermarks, {totals['borders']} with borders. Mean words: long {mean['long']:.0f}, "
              f"medium {mean['medium']:.0f}, short {mean['short']:.0f}.", "",
              "| id | short | words l/m/s | style words | wm | borders | problems |", "|---|---|---|---|---|---|---|"]
    report += ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in row) + " |" for row in rows]
    out = work / f"review_{args.prompt}.md"
    write_text(out, "\n".join(report) + "\n")
    log(report[2])
    log(f"report: {out}")


# ---------------------------------------------------------------- build


def clean_line(text: str) -> str:
    text = " ".join(text.split())  # one line: with enable_wildcard every line is a separate alternative
    if "|||" in text:
        raise ValueError("'|||' in a caption")
    return text.replace("{", "{{").replace("}", "}}")  # literal braces under enable_wildcard


def cmd_build(args) -> None:
    work = Path(args.work).resolve()
    layout = dict(zip(("original",) + FIELDS, (int(v) for v in args.layout.split(","))))
    if len(layout) != 4 or layout["original"] < 1:
        sys.exit("--layout needs 4 counts: original,short,medium,long (original >= 1)")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = work / f"_backup_captions_{stamp}"
    progress = work / "progress_build.jsonl"
    plan, missing = [], []
    for subset in work_subsets(work):
        for name in subset["images"]:
            stem = Path(name).stem
            target = work / subset["name"] / (stem + subset["caption_extension"])
            original = work / ORIG / subset["name"] / (stem + subset["caption_extension"])
            lines = [line.strip() for line in read_text(original).splitlines() if line.strip()] if original.is_file() else []
            source = output_path(work / subset["name"] / name, args.prompt)
            if not source.is_file():
                missing.append(f"{subset['name']}/{stem}")
                plan.append((target, lines))  # the original caption only
                continue
            caption = json.loads(read_text(source))["caption"]
            recaps = [f"{args.trigger} ||| ||| {clean_line(caption[f])}" for f in FIELDS for _ in range(layout[f])]
            plan.append((target, lines * layout["original"] + recaps))

    if missing:
        log(f"{len(missing)} images have no {args.prompt} JSON and keep their original caption: {', '.join(missing[:10])}"
            + (" ..." if len(missing) > 10 else ""))
    example = next((p for p in plan if len(p[1]) > 1), None)
    if example:
        log(f"example {example[0]}:\n  " + "\n  ".join(line[:160] for line in example[1]))
    share = layout["original"] / sum(layout.values())
    log(f"{len(plan)} caption files; layout {args.layout} (with cache_caption_variants = 16 about "
        f"{(1 + 15 * share) / 16:.0%} original captions)")
    if not args.apply:
        log("dry run: nothing written (add --apply)")
        return
    for target, lines in plan:
        if target.is_file():
            destination = backup / target.relative_to(work)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, destination)
        write_text(target, "\n".join(lines) + "\n")
        append_jsonl(progress, {"file": str(target), "lines": len(lines), "prompt": args.prompt, "at": stamp})
    log(f"wrote {len(plan)} caption files in {work}; previous ones in {backup}")


# ---------------------------------------------------------------- main


def main() -> None:
    sys.stdout.reconfigure(errors="replace")  # a cp949 console must not stop a run over one path character
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("copy", help="copy a dataset config's subsets into a work folder")
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--apply", action="store_true")
    p.set_defaults(func=cmd_copy)

    p = sub.add_parser("queue", help="list the images and their reference captions")
    p.add_argument("--work", required=True)
    p.add_argument("--trigger", required=True, help="removed from the reference caption (also any @handle)")
    p.add_argument("--pilot", type=int, default=0, help="also pick this many images (stratified by subset)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repick", action="store_true")
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("run", help="caption with Codex CLI (ChatGPT plan), one call per image")
    p.add_argument("--work", required=True)
    p.add_argument("--queue", default="pilot.jsonl", help="queue file in the work folder (pilot.jsonl or queue.jsonl)")
    p.add_argument("--prompt", choices=sorted(PROMPTS), default="official")
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--effort", default="high", help="model_reasoning_effort")
    p.add_argument("--sandbox", default="read-only", help="codex exec -s (the model needs no tools)")
    p.add_argument("--codex", help="path to codex.exe (default: the newest under %%LOCALAPPDATA%%\\OpenAI\\Codex\\bin)")
    p.add_argument("--timeout", type=int, default=900, help="seconds per image")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--redo", action="store_true", help="caption again images that already have a JSON")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check", help="validate the JSON files and write review_<prompt>.md")
    p.add_argument("--work", required=True)
    p.add_argument("--prompt", choices=sorted(PROMPTS), default="official")
    p.add_argument("--trigger", default="")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("build", help="write the multi-line training captions in the work folder")
    p.add_argument("--work", required=True)
    p.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    p.add_argument("--trigger", required=True)
    p.add_argument("--layout", default="1,2,2,2", help="lines of original,short,medium,long")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
