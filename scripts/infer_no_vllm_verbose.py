import os, json, time, argparse, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def log(msg): print(f"[{time.strftime('%F %T')}] {msg}", flush=True)

def load_ckpt_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt

def read_jsonl(path, n=-1):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n > 0 and i >= n: break
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

def pick_keys(x):
    if "input" in x and "output" in x: return "input", "output"
    if "problem" in x and "answer" in x: return "problem", "answer"
    raise ValueError(f"Unknown keys: {list(x.keys())}")

def fmt(sec):
    sec = int(max(0, sec)); m, s = divmod(sec, 60); h, m = divmod(m, 60)
    return f"{h}h{m}m{s}s" if h else f"{m}m{s}s"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--ckpt", default="")  # #Edit
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--read_num", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)  # #Edit
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"device={device}, cuda_visible={os.getenv('CUDA_VISIBLE_DEVICES')}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True
    ).eval()

    if args.ckpt and os.path.isfile(args.ckpt):
        state = load_ckpt_state(args.ckpt)
    else:
        log("No valid ckpt file provided, using base model only.")  # #Edit

    data = read_jsonl(args.dataset, args.read_num)
    if not data: raise ValueError("Dataset is empty")
    in_key, out_key = pick_keys(data[0])
    total = len(data)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    t0 = time.time()
    done = 0
    with open(args.save_path, "w", encoding="utf-8") as w:
        pbar = tqdm(total=total, desc="Infer", ncols=100)
        for st in range(0, total, args.batch_size):  # #Edit
            batch = data[st:st + args.batch_size]
            prompts = [f"Question:\n{x[in_key]}\n\nAnswer:" for x in batch]
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

            in_lens = inputs["attention_mask"].sum(dim=1).tolist()
            for i, x in enumerate(batch):
                pred = tok.decode(out[i][in_lens[i]:], skip_special_tokens=True)
                w.write(json.dumps({"question": x[in_key], "gold": x[out_key], "pred": pred}, ensure_ascii=False) + "\n")

            done += len(batch)
            pbar.update(len(batch))
            if done % 10 == 0 or done == total:
                avg = (time.time() - t0) / done
                eta = avg * (total - done)
                log(f"Progress {done}/{total} | avg={avg:.2f}s/item | ETA={fmt(eta)}")
        pbar.close()

    log(f"Done: {args.save_path}")

if __name__ == "__main__":
    main()
