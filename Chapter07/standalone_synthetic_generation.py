#!/usr/bin/env python3
"""
Standalone CUAD synthetic dataset builder with rationale generation and Hub upload.

This single-file script performs the complete workflow:
1) Download the CUAD dataset from Zenodo (CUAD_v1.zip) and extract it.
2) Convert CUAD SQuAD-style annotations into a simplified JSONL format with optional
   negative "NONE" examples and surrounding context for each clause.
3) Split the dataset into train/validation/test JSONL files.
4) Optionally generate model rationales for a chosen split or for all splits using
   DeepSeek models served via OpenRouter (OpenAI-compatible API).
5) Optionally push the base dataset and/or the rationales dataset to the Hugging Face Hub.

Environment variables expected:
- HF_API_KEY: Hugging Face token used to authenticate "push_to_hub" operations.
- DEEPSEEK_BOOK_OPENROUTER_API_KEY: OpenRouter API key used to call DeepSeek models.

Assumed installed packages (install via pip if missing):
- requests
- datasets
- huggingface_hub
- scikit-learn
- rich
- openai>=1.0.0  (for OpenAI-compatible API on OpenRouter)
- backoff

Example Usages:
- Run the base pipeline (download -> process -> split):
  python standalone_synthetic_generation.py

- Also generate rationales for the test split and push to the Hub:
  python standalone_synthetic_generation.py --generate-rationales --rationale-target test --push-rationales-to-hub --rationales-repo-id your-username/cuad-deepseek-rationales

- End-to-end everything (download, process, split, rationales for all splits, push):
  python standalone_synthetic_generation.py --run-all --rationale-target all --base-repo-id your-username/cuad-simplified --rationales-repo-id your-username/cuad-deepseek-rationales

Notes:
- By default, rationales are generated for the "test" split only to minimize costs.
- Use --rationale-target all to annotate all splits (train/validation/test).
- The script is idempotent where possible and supports resume for rationale generation.
"""

import argparse
import concurrent.futures
import json
import os
import random
import threading
import time
import zipfile
from collections import Counter
from datetime import datetime
from statistics import mean, median
from typing import Dict, List, Optional, Set, Tuple

import backoff
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sklearn.model_selection import train_test_split

console = Console()

# -----------------------------
# Section 1: Download CUAD
# -----------------------------

DEFAULT_ZENODO_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"


def download_and_extract_cuad(
    zenodo_url: str,
    data_dir: str = "data",
    force: bool = False,
) -> str:
    """
    Download and extract CUAD_v1.zip from Zenodo.

    Why this design:
    - Stream download with progress for reliability and UX.
    - Extract into data_dir so paths remain stable across environments.
    - Idempotent by default; force=True to re-download.

    Returns:
        Path to the extracted CUAD directory, typically data/CUAD_v1
    """
    os.makedirs(data_dir, exist_ok=True)
    expected_dir = os.path.join(data_dir, "CUAD_v1")
    expected_json = os.path.join(expected_dir, "CUAD_v1.json")
    zip_path = os.path.join(data_dir, "CUAD_v1.zip")

    if os.path.exists(expected_json) and not force:
        console.print(f"[green]CUAD already present at {expected_json}. Skipping download.[/green]")
        return expected_dir

    # Download the zip with streaming and progress
    console.print(f"[blue]Downloading CUAD from {zenodo_url}[/blue]")
    with requests.get(zenodo_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = 1024 * 64

        with open(zip_path, "wb") as f, Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Downloading CUAD_v1.zip...", total=total if total else None)
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
                    if total:
                        progress.update(task, advance=len(part))

    # Extract to data_dir
    console.print(f"[blue]Extracting to {data_dir}...[/blue]")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    console.print(f"[green]CUAD extracted to {expected_dir}[/green]")
    return expected_dir


# -----------------------------
# Section 2: Simplify CUAD -> JSONL with optional negatives
# -----------------------------

def load_cuad_squad_json(base_path: str) -> Dict:
    """
    Load the SQuAD-style JSON shipped with CUAD.

    Returns:
        Parsed JSON dict.
    """
    squad_path = os.path.join(base_path, "CUAD_v1.json")
    if not os.path.exists(squad_path):
        raise FileNotFoundError(f"Could not find CUAD_v1.json at {squad_path}")
    with open(squad_path, "r") as f:
        return json.load(f)


def extract_clauses_from_squad(
    squad_data: Dict,
    context_window: int = 150,
    include_negatives: bool = False,
    negative_ratio: float = 3.0,
    negative_clause_length: int = 100,
) -> List[Dict]:
    """
    Convert CUAD SQuAD JSON into a list of clause records for JSONL.

    Why this design:
    - Preserve core fields: clause, context, label, contract metadata.
    - Optionally mine "NONE" negatives from non-annotated spans to improve robustness.
    - Keep deterministic behavior with a fixed random seed.

    Negative example logic:
    - Track labeled spans; identify gaps; sample fixed-length chunks from gaps.
    - Limit # negatives per gap to avoid dominating positives.

    Returns:
        List of dictionaries with clause text, context, label and metadata.
    """
    clauses: List[Dict] = []

    # Track labeled regions for negative sampling
    labeled_regions = {}
    if include_negatives:
        labeled_regions = {}

    for doc in squad_data["data"]:
        contract_name = doc.get("title", "")
        contract_type = "Unknown"
        if contract_name and "-" in contract_name:
            parts = contract_name.split("-")
            if len(parts) >= 3:
                contract_type = parts[-1].strip()

        if include_negatives and contract_name not in labeled_regions:
            labeled_regions[contract_name] = {}

        for para_idx, paragraph in enumerate(doc.get("paragraphs", [])):
            para_text = paragraph.get("context", "")
            if not para_text.strip():
                continue

            if include_negatives and para_idx not in labeled_regions.get(contract_name, {}):
                labeled_regions[contract_name][para_idx] = []

            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible", False):
                    continue

                # Format often is "CONTRACTNAME__Category"
                category = qa.get("id", "").split("__")[-1] if qa.get("id") else "UNKNOWN"

                for answer in qa.get("answers", []):
                    answer_text = answer.get("text", "")
                    answer_start = answer.get("answer_start", 0)
                    answer_end = answer_start + len(answer_text)

                    if include_negatives:
                        labeled_regions[contract_name][para_idx].append((answer_start, answer_end))

                    # Clause and context
                    clause_text = answer_text
                    context_start = max(0, answer_start - context_window)
                    context_end = min(len(para_text), answer_end + context_window)
                    context_text = para_text[context_start:context_end]

                    clauses.append(
                        {
                            "clause": clause_text,
                            "clause_with_context": context_text,
                            "label": category,
                            "contract_name": contract_name,
                            "contract_type": contract_type,
                        }
                    )

    # Generate negatives if requested
    if include_negatives:
        import random as _random  # local random alias for clarity
        n_positives = len(clauses)
        n_negatives_to_generate = int(n_positives * negative_ratio)
        console.print(f"Generating [bold]{n_negatives_to_generate}[/bold] negative examples...")

        negatives: List[Dict] = []
        _random.seed(42)

        for doc in squad_data["data"]:
            contract_name = doc.get("title", "")
            contract_type = "Unknown"
            if contract_name and "-" in contract_name:
                parts = contract_name.split("-")
                if len(parts) >= 3:
                    contract_type = parts[-1].strip()

            for para_idx, paragraph in enumerate(doc.get("paragraphs", [])):
                para_text = paragraph.get("context", "")
                if not para_text.strip():
                    continue

                labeled_spans = labeled_regions.get(contract_name, {}).get(para_idx, [])
                labeled_spans.sort(key=lambda x: x[0])

                gaps = []
                last_end = 0
                for start, end in labeled_spans:
                    if start > last_end + negative_clause_length:
                        gaps.append((last_end, start))
                    last_end = max(last_end, end)

                if len(para_text) > last_end + negative_clause_length:
                    gaps.append((last_end, len(para_text)))

                for gap_start, gap_end in gaps:
                    gap_length = gap_end - gap_start
                    if gap_length < negative_clause_length:
                        continue

                    n_examples = min(3, gap_length // negative_clause_length)
                    for _ in range(n_examples):
                        if len(negatives) >= n_negatives_to_generate:
                            break
                        max_start = gap_end - negative_clause_length
                        if gap_start >= max_start:
                            continue
                        start = _random.randint(gap_start, max_start)
                        neg_clause = para_text[start : start + negative_clause_length]

                        context_start = max(0, start - context_window)
                        context_end = min(len(para_text), start + negative_clause_length + context_window)
                        context_text = para_text[context_start:context_end]

                        negatives.append(
                            {
                                "clause": neg_clause,
                                "clause_with_context": context_text,
                                "label": "NONE",
                                "contract_name": contract_name,
                                "contract_type": contract_type,
                            }
                        )

                if len(negatives) >= n_negatives_to_generate:
                    break
            if len(negatives) >= n_negatives_to_generate:
                break

        num_added = min(len(negatives), n_negatives_to_generate)
        clauses.extend(negatives[:num_added])
        console.print(f"Added [bold green]{num_added}[/bold green] negative examples with label 'NONE'")

    return clauses


def save_jsonl(data: List[Dict], output_path: str) -> None:
    """Write list of dicts to JSONL, one item per line."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_jsonl(file_path: str) -> List[Dict]:
    """Read a JSONL file into a list of dicts."""
    out: List[Dict] = []
    with open(file_path, "r") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def create_data_splits(
    data: List[Dict],
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Stratified split into train/val/test with fallback if any class is too small.

    Why this design:
    - Prefer stratified splits to preserve label distribution.
    - Fall back gracefully if a class lacks sufficient samples for stratification.
    """
    # First split: train vs (val+test)
    train_data, temp_data = train_test_split(
        data,
        train_size=train_size,
        random_state=random_state,
        stratify=[item["label"] for item in data],
    )

    # Compute relative val size inside temp portion
    relative_val_size = val_size / (1.0 - train_size)

    # Check if temp split can be stratified
    temp_labels = [item["label"] for item in temp_data]
    label_counts = Counter(temp_labels)
    if any(count < 2 for count in label_counts.values()):
        val_data, test_data = train_test_split(
            temp_data,
            train_size=relative_val_size,
            random_state=random_state,
        )
    else:
        val_data, test_data = train_test_split(
            temp_data,
            train_size=relative_val_size,
            random_state=random_state,
            stratify=temp_labels,
        )

    return train_data, val_data, test_data


def to_hf_dataset(data: List[Dict] = None, jsonl_path: str = None) -> Dataset:
    """Build a Hugging Face Dataset either from a list of dicts or an on-disk JSONL."""
    if data is not None:
        return Dataset.from_list(data)
    if jsonl_path is not None:
        return Dataset.from_list(load_jsonl(jsonl_path))
    raise ValueError("Either data or jsonl_path must be provided")


def push_to_hub(dataset_or_dict, repo_id: str, private: bool = False) -> None:
    """
    Push a Hugging Face Dataset or DatasetDict to the Hub.

    Why this design:
    - Auth via HF_API_KEY environment variable avoids interactive login.
    - Leave visibility choice to caller.
    """
    token = os.environ.get("HF_API_KEY")
    if not token:
        console.print("[bold red]HF_API_KEY environment variable not found.[/bold red]")
        console.print("Set HF_API_KEY to your Hugging Face token.")
        return

    try:
        login(token=token)
        console.print(f"[blue]Pushing dataset to HF Hub: {repo_id}[/blue]")
        dataset_or_dict.push_to_hub(repo_id=repo_id, private=private)
        console.print(f"[green]Successfully pushed to {repo_id}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error pushing to Hugging Face Hub: {e}[/bold red]")


# -----------------------------
# Section 3: Rationale generation via OpenRouter (DeepSeek)
# -----------------------------

OPENROUTER_API_KEY = os.getenv("DEEPSEEK_BOOK_OPENROUTER_API_KEY")


def create_prompt(sample: Dict) -> str:
    """Construct the rationale prompt for a single clause sample."""
    return f"""[[TASK]]
Analyze the contract clause below and provide a rationale for its classification.

[[CONTEXT]]
CLAUSE: {sample["clause"]}

CLAUSE WITH CONTEXT: {sample["clause_with_context"]}

CONTRACT TYPE: {sample["contract_type"]}

CLASSIFICATION: {sample["label"]}

[[INSTRUCTIONS]]
Explain WHY this clause was classified as '{sample["label"]}'. Identify specific language or elements in the clause that led to this categorization. If the classification is "NONE", explain why it doesn't fall into any contract category.

Take your time to think through all aspects carefully. Consider:
1. Key terms and phrases in the clause
2. Function of this clause in a contract
3. How it relates to the contract type
4. Why it fits or doesn't fit specific categories

[[FORMAT]]
Your response should be:
- Concise
- Focus on specific clause language that justifies the classification
- Include legal reasoning where appropriate
- Formatted with Markdown for readability
"""


class RateLimiter:
    """
    Simple thread-safe rate limiter to avoid exceeding API QPS.

    Why this design:
    - Multi-threaded generation benefits from a shared limiter to smooth bursts.
    - Jitter reduces synchronized call spikes across threads.
    """
    def __init__(self, calls_per_second: float):
        self.calls_per_second = max(float(calls_per_second), 0.1)
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            since_last = now - self.last_call_time
            min_interval = (1.0 / self.calls_per_second) + random.uniform(0, 0.2)
            if since_last < min_interval:
                time.sleep(min_interval - since_last)
            self.last_call_time = time.time()


# Backoff wrapper for API calls
@backoff.on_exception(
    backoff.expo,
    (Exception,),
    max_tries=5,
    max_time=300,
    on_backoff=lambda details: console.print(
        f"[yellow]API call failed. Retrying in {details['wait']:.1f}s (attempt {details['tries']}/5)...[/yellow]"
    ),
)
def generate_rationale_once(
    client: OpenAI,
    prompt: str,
    limiter: RateLimiter,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call OpenRouter Chat Completions API with exponential backoff.

    Returns:
        (content, reasoning_trace) - reasoning_trace may be None if not provided by the model.
    """
    limiter.wait()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    message = completion.choices[0].message
    content = message.content
    reasoning = getattr(message, "reasoning", None)
    return content, reasoning


class ProcessingStats:
    """Track per-example processing times across threads for ETA calculations."""
    def __init__(self):
        self.processing_times: List[float] = []
        self.lock = threading.Lock()

    def add(self, seconds: float) -> None:
        with self.lock:
            self.processing_times.append(seconds)

    def avg(self) -> float:
        with self.lock:
            return mean(self.processing_times) if self.processing_times else 0.0

    def med(self) -> float:
        with self.lock:
            return median(self.processing_times) if self.processing_times else 0.0


def get_processed_indices(output_path: str) -> Set[int]:
    """
    Scan an existing output JSONL and return the set of indices already present.

    Why this design:
    - Enables safe resume after interrupts or failures by skipping completed items.
    """
    processed: Set[int] = set()
    if not os.path.exists(output_path):
        return processed
    try:
        with open(output_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                if "index" in obj:
                    processed.add(int(obj["index"]))
    except Exception:
        console.print(f"[yellow]Warning: Could not read processed indices from {output_path}[/yellow]")
    return processed


def append_jsonl_threadsafe(path: str, item: Dict], lock: threading.Lock) -> bool:
    """Append a dict to JSONL with a file-level lock to avoid interleaved writes."""
    try:
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(item) + "\n")
        return True
    except Exception as e:
        console.print(f"[bold red]Error writing to {path}: {e}[/bold red]")
        return False


def generate_rationales_for_jsonl(
    input_path: str,
    output_path: str,
    log_path: str,
    model: str = "deepseek/deepseek-r1",
    max_tokens: int = 1500,
    temperature: float = 0.2,
    top_p: float = 0.95,
    workers: int = 10,
    rate_limit: float = 10.0,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Generate rationales for each example in a JSONL file and append results to another JSONL.

    Why this design:
    - Uses a thread pool to parallelize calls while honoring a global rate limit.
    - Resumable by skipping indices already present in the output file.

    Returns:
        (completed_count, failed_count)
    """
    # Guard: API key
    if not OPENROUTER_API_KEY:
        console.print("[bold red]DEEPSEEK_BOOK_OPENROUTER_API_KEY not set. Skipping rationale generation.[/bold red]")
        return (0, 0)

    # Load input
    try:
        data = load_jsonl(input_path)
    except Exception as e:
        console.print(f"[bold red]Failed to load {input_path}: {e}[/bold red]")
        return (0, 0)

    if not data:
        console.print(f"[yellow]No records found in {input_path}. Nothing to process.[/yellow]")
        return (0, 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize client and shared state
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    limiter = RateLimiter(rate_limit)
    file_lock = threading.Lock()
    stats = ProcessingStats()

    processed_indices = get_processed_indices(output_path)
    todo_indices = [i for i in range(len(data)) if i not in processed_indices]
    total_to_process = len(todo_indices)

    if total_to_process == 0:
        console.print(f"[green]All examples in {input_path} already have rationales in {output_path}[/green]")
        return (0, 0)

    console.print(
        f"[blue]Generating rationales for {total_to_process} examples "
        f"(model={model}, workers={workers}, QPS~{rate_limit})[/blue]"
    )

    # Prepare log
    with open(log_path, "w") as log:
        log.write(f"Started: {datetime.now().isoformat()}\n")
        log.write(f"Input: {input_path}\n")
        log.write(f"Output: {output_path}\n")
        log.write(f"Model: {model}\n")
        log.write(f"Temperature: {temperature}, Top-p: {top_p}\n")
        log.write(f"Workers: {workers}, Rate limit: {rate_limit}\n")
        log.write(f"Remaining examples: {total_to_process}\n")

    completed = 0
    failed = 0

    class ProgressTracker:
        """Minimal wrapper to coordinate progress updates and ETA messaging."""
        def __init__(self, progress_bar, task_id):
            self.progress_bar = progress_bar
            self.task_id = task_id
            self.lock = threading.Lock()

        def update(self):
            with self.lock:
                self.progress_bar.update(self.task_id, advance=1)
                # Median time is more stable for ETA with outliers
                med = stats.med()
                if med > 0:
                    done = self.progress_bar.tasks[self.task_id].completed
                    remaining = total_to_process - done
                    effective_workers = min(workers, max(remaining, 1))
                    eta_sec = (med * remaining) / max(effective_workers, 1)
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
                    self.progress_bar.update(
                        self.task_id,
                        description=f"[cyan]Generating rationales (ETA {eta_str}, ~{med:.1f}s/ea)",
                    )

    def process_one(idx: int) -> bool:
        """Worker routine for a single example."""
        start = time.time()
        sample = data[idx]
        try:
            prompt = create_prompt(sample)
            content, reasoning = generate_rationale_once(
                client=client,
                prompt=prompt,
                limiter=limiter,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            result = {
                **sample,
                "rationale": content,
                "reasoning_trace": reasoning,
                "index": idx,
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
            }
            ok = append_jsonl_threadsafe(output_path, result, file_lock)
            if ok and verbose:
                with file_lock:
                    preview = f"[bold]Label:[/bold] {sample.get('label')}  |  [bold]Rationale:[/bold] {(content or '')[:150]}..."
                    if reasoning:
                        preview += f"\n[bold]Reasoning trace:[/bold] {reasoning[:150]}..."
                    console.print(Panel(preview, title=f"Example {idx}", border_style="green"))
            return ok
        except Exception as e:
            with file_lock:
                console.print(f"[red]Error on idx={idx}: {e}[/red]")
            return False
        finally:
            stats.add(time.time() - start)

    # Progress UI + thread pool
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[cyan]Generating rationales...", total=total_to_process)
        tracker = ProgressTracker(progress, task_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_one, idx): idx for idx in todo_indices}
            for fut in concurrent.futures.as_completed(futures):
                ok = False
                try:
                    ok = fut.result()
                except Exception as e:
                    console.print(f"[red]Unhandled error: {e}[/red]")
                    ok = False
                if ok:
                    completed += 1
                else:
                    failed += 1
                tracker.update()

    # Summary
    table = Table(title="Rationale Generation Summary", box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total in input", str(len(data)))
    table.add_row("Already processed (skipped)", str(len(processed_indices)))
    table.add_row("Attempted in this run", str(total_to_process))
    table.add_row("Completed", str(completed))
    table.add_row("Failed", str(failed))
    table.add_row("Average time/example", f"{stats.avg():.2f}s")
    table.add_row("Median time/example", f"{stats.med():.2f}s")
    console.print(table)

    with open(log_path, "a") as log:
        log.write(f"Completed: {datetime.now().isoformat()}\n")
        log.write(f"Completed this run: {completed}, Failed: {failed}\n")

    return completed, failed


# -----------------------------
# Section 4: Orchestration (end-to-end)
# -----------------------------

def build_base_dataset(
    cuad_dir: str,
    output_dir: str = "data/processed_cuad",
    context_window: int = 150,
    include_negatives: bool = False,
    negative_ratio: float = 3.0,
    negative_clause_length: int = 100,
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_seed: int = 42,
    do_splits: bool = True,
) -> Dict[str, str]:
    """
    Create simplified JSONL from CUAD and optional splits; returns file paths.

    Returns dict with keys:
      - full
      - train (if do_splits)
      - validation (if do_splits)
      - test (if do_splits)
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, "full_data.jsonl")
    train_path = os.path.join(output_dir, "train_data.jsonl")
    val_path = os.path.join(output_dir, "val_data.jsonl")
    test_path = os.path.join(output_dir, "test_data.jsonl")

    console.print(f"[blue]Loading CUAD from {cuad_dir}[/blue]")
    squad = load_cuad_squad_json(cuad_dir)

    console.print("[blue]Extracting clauses and labels...[/blue]")
    items = extract_clauses_from_squad(
        squad_data=squad,
        context_window=context_window,
        include_negatives=include_negatives,
        negative_ratio=negative_ratio,
        negative_clause_length=negative_clause_length,
    )

    pos = sum(1 for x in items if x["label"] != "NONE")
    neg = sum(1 for x in items if x["label"] == "NONE")
    console.print(f"[green]Extracted {len(items)} total clauses ({pos} positive, {neg} 'NONE')[/green]")
    console.print(f"[green]Context window: {context_window} chars[/green]")

    # Save full
    save_jsonl(items, full_path)
    console.print(f"[green]Saved full dataset to {full_path}[/green]")

    paths = {"full": full_path}

    if do_splits:
        console.print(
            f"[blue]Creating splits: train={train_size:.0%}, val={val_size:.0%}, test={(1.0-train_size-val_size):.0%}[/blue]"
        )
        train_data, val_data, test_data = create_data_splits(
            items, train_size=train_size, val_size=val_size, random_state=random_seed
        )
        save_jsonl(train_data, train_path)
        save_jsonl(val_data, val_path)
        save_jsonl(test_data, test_path)
        console.print(f"[green]Saved splits to {output_dir}[/green]")

        # Basic split info
        info = Table(title="Split Summary", box=None)
        info.add_column("Split", style="cyan")
        info.add_column("#Items", style="magenta")
        info.add_row("train", str(len(train_data)))
        info.add_row("validation", str(len(val_data)))
        info.add_row("test", str(len(test_data)))
        console.print(info)

        paths.update({"train": train_path, "validation": val_path, "test": test_path})

    return paths


def push_jsonl_splits_to_hub(
    split_paths: Dict[str, str],
    repo_id: str,
    private: bool = False,
) -> None:
    """
    Load split JSONLs into a DatasetDict and push to Hugging Face Hub.

    Why this design:
    - Avoid accidental pushing of partial sets by requiring a dict of split paths.
    """
    datasets = {}
    for split, path in split_paths.items():
        datasets[split] = to_hf_dataset(jsonl_path=path)

    dd = DatasetDict(datasets)
    push_to_hub(dd, repo_id=repo_id, private=private)


def run_pipeline(args: argparse.Namespace) -> None:
    """Orchestrate the end-to-end process based on CLI flags."""
    # 1) Download CUAD if needed
    if not args.skip_download:
        cuad_dir = download_and_extract_cuad(
            zenodo_url=args.cuad_url,
            data_dir=args.data_dir,
            force=args.force_download,
        )
    else:
        cuad_dir = os.path.join(args.data_dir, "CUAD_v1")
        if not os.path.exists(os.path.join(cuad_dir, "CUAD_v1.json")):
            raise FileNotFoundError(
                f"--skip-download specified but {os.path.join(cuad_dir, 'CUAD_v1.json')} not found."
            )
        console.print(f"[green]Using existing CUAD at {cuad_dir}[/green]")

    # 2) Build simplified dataset + splits
    split_paths = build_base_dataset(
        cuad_dir=cuad_dir,
        output_dir=args.output_dir,
        context_window=args.context_window,
        include_negatives=args.include_negatives,
        negative_ratio=args.negative_ratio,
        negative_clause_length=args.negative_clause_length,
        train_size=args.train_size,
        val_size=args.val_size,
        random_seed=args.random_seed,
        do_splits=not args.no_split,
    )

    # 3) Optionally push base dataset
    if args.push_base_to_hub:
        # Build DatasetDict for push (prefer splits if available, else full only)
        if "train" in split_paths and "validation" in split_paths and "test" in split_paths:
            base_splits = {
                "train": split_paths["train"],
                "validation": split_paths["validation"],
                "test": split_paths["test"],
            }
            console.print("[blue]Pushing base dataset splits to Hub...[/blue]")
            push_jsonl_splits_to_hub(base_splits, repo_id=args.base_repo_id, private=args.private)
        else:
            # Only full available
            dd = DatasetDict({"full": to_hf_dataset(jsonl_path=split_paths["full"])})
            push_to_hub(dd, repo_id=args.base_repo_id, private=args.private)

    # 4) Optionally generate rationales
    if args.generate_rationales:
        targets: List[str]
        if args.rationale_target == "all":
            # Require or ensure splits exist
            if not all(k in split_paths for k in ("train", "validation", "test")):
                raise ValueError("rationale-target=all requires dataset splits. Disable --no-split if enabled.")
            targets = ["train", "validation", "test"]
        elif args.rationale_target in ("full", "train", "validation", "test"):
            # Validate presence
            if args.rationale_target not in split_paths:
                raise ValueError(f"Target split '{args.rationale_target}' not available.")
            targets = [args.rationale_target]
        else:
            raise ValueError(f"Unknown rationale target: {args.rationale_target}")

        rationale_paths: Dict[str, str] = {}
        for target in targets:
            in_path = split_paths[target]
            stem = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(args.output_dir, f"{stem}_with_rationales.jsonl")
            log_path = os.path.join(args.output_dir, f"rationale_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

            completed, failed = generate_rationales_for_jsonl(
                input_path=in_path,
                output_path=out_path,
                log_path=log_path,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                workers=args.workers,
                rate_limit=args.rate_limit,
                verbose=args.verbose,
            )
            console.print(
                f"[green]{target}: completed {completed}, failed {failed}. Output -> {out_path}[/green]"
            )
            rationale_paths[target] = out_path

        # 5) Optionally push rationales dataset
        if args.push_rationales_to_hub:
            # If only single target was generated, still push as DatasetDict with one split.
            console.print("[blue]Pushing rationales dataset to Hub...[/blue]")
            push_jsonl_splits_to_hub(rationale_paths, repo_id=args.rationales_repo_id, private=args.private)

    console.print("[bold green]Pipeline complete.[/bold green]")


# -----------------------------
# Section 5: CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone CUAD synthetic dataset + rationales + HF Hub upload"
    )

    # Download settings
    p.add_argument("--data-dir", default="data", help="Base directory for raw data (CUAD will be extracted here).")
    p.add_argument("--cuad-url", default=DEFAULT_ZENODO_URL, help="Zenodo URL for CUAD_v1.zip.")
    p.add_argument("--skip-download", action="store_true", help="Skip CUAD download if already present.")
    p.add_argument("--force-download", action="store_true", help="Force re-download and re-extract CUAD.")

    # Processing / simplification settings
    p.add_argument("--output-dir", default="data/processed_cuad", help="Directory to write processed JSONL files.")
    p.add_argument("--context-window", type=int, default=150, help="Chars of context to include around each clause.")
    p.add_argument("--include-negatives", action="store_true", help="Include synthetic 'NONE' negative examples.")
    p.add_argument("--negative-ratio", type=float, default=3.0, help="Negatives per positive ratio.")
    p.add_argument("--negative-clause-length", type=int, default=100, help="Approx length of negative 'clause' chunks.")
    p.add_argument("--train-size", type=float, default=0.8, help="Train split proportion.")
    p.add_argument("--val-size", type=float, default=0.1, help="Validation split proportion.")
    p.add_argument("--random-seed", type=int, default=42, help="Random seed for splits.")
    p.add_argument("--no-split", action="store_true", help="Do not create train/val/test splits (keep full only).")

    # Rationale generation settings
    p.add_argument("--generate-rationales", action="store_true", help="Generate rationales with DeepSeek via OpenRouter.")
    p.add_argument(
        "--rationale-target",
        choices=["full", "train", "validation", "test", "all"],
        default="test",
        help="Which part(s) to annotate with rationales. Default: test",
    )
    p.add_argument("--model", default="deepseek/deepseek-r1", help="Model name for OpenRouter.")
    p.add_argument("--max-tokens", type=int, default=1500, help="Max tokens for rationale generation.")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    p.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    p.add_argument("--workers", type=int, default=10, help="Parallel worker threads.")
    p.add_argument("--rate-limit", type=float, default=10.0, help="Max calls per second across all workers.")
    p.add_argument("--verbose", action="store_true", help="Verbose output with rationale previews.")

    # Hub push settings
    p.add_argument("--push-base-to-hub", action="store_true", help="Push the base (no-rationale) dataset to HF Hub.")
    p.add_argument("--push-rationales-to-hub", action="store_true", help="Push the rationales dataset to HF Hub.")
    p.add_argument("--base-repo-id", default="strickvl/cuad-deepseek", help="HF Hub repo for base dataset.")
    p.add_argument("--rationales-repo-id", default="zenml/cuad-deepseek", help="HF Hub repo for rationales dataset.")
    p.add_argument("--private", action="store_true", help="Create private repos on HF Hub.")

    # Convenience: run all stages
    p.add_argument(
        "--run-all",
        action="store_true",
        help="Do everything: download, process, split, generate rationales for all splits, and push rationales to Hub.",
    )
    return p


def apply_run_all_defaults(args: argparse.Namespace) -> None:
    """
    Mutate args to sensible 'run all' settings if --run-all is provided.

    Why this design:
    - Avoid surprising defaults when user wants a full end-to-end run.
    """
    if args.run_all:
        args.skip_download = False if not args.skip_download else args.skip_download
        args.generate_rationales = True
        args.rationale_target = "all"
        args.push_rationales_to_hub = True
        # Ensure splits exist for 'all'
        args.no_split = False


def main():
    args = build_arg_parser().parse_args()
    apply_run_all_defaults(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()
