# Annotated Transformer – Debugging & Setup Notes

## 1. Environment Setup

Run these commands in the terminal to install dependencies inside a new conda environment.

```bash
conda create --name genai python=3.11
conda activate genai

pip install torch==2.0.1 torchtext==0.15.2 torchdata==0.6.1 spacy==3.6.1 pandas==1.5.3
pip install altair GPUtil ipykernel numpy==1.26.4
```

### If NumPy issues arise

```bash
pip uninstall -y numpy
pip install numpy==1.26.4
```

---

## 2. Install spaCy Models

IF the models are not getting downloaded or showing error in the cell where the toekinzers are being loaded, this is the fix. 

DO Check the release links for spaCy models compatible with version **3.6.0**. if the ones below doesn't work.


```bash
pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.6.0/de_core_news_sm-3.6.0-py3-none-any.whl

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl
```

---

## 4. Verify Environment Versions

After imports, run:

```python
from torchtext import __version__ as torchtext_version
from torchdata import __version__ as torchdata_version

print(f"Python version: {os.sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"torchtext version: {torchtext_version}")
print(f"torchdata version: {torchdata_version}")
print(f"Spacy version: {spacy.__version__}")
```

---

## 5. Fix Broken Multi30k URLs

this shows up as SSL-ish named error. 

Add after imports:

```python
from torchtext.datasets import multi30k

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
multi30k.URL["test"]  = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"

multi30k.MD5 = {
    "train": "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e",
    "valid": "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c",
    "test":  "6d1ca1dba99e2c5dd54cae1226ff11c2551e6ce63527ebb072a1f70f72a5cd36",
}
```

---

## 6. UTF-8 Encoding Error During Vocabulary Building

If German vocabulary building fails with encoding errors:

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80
```

### Steps

1. Go to:

```
C:\Users\<USER_NAME>\.cache\torch\text\datasets\Multi30k
```

2. Extract:

```
mmt16_task1_test.tar.gz
```

3. After extraction you should see:

```
test.de
test.en
test.fr
._test.de
._test.en
._test.fr
```

Replace the corrupted files in the same directory.

4. The files:

```
._test.de
._test.en
._test.fr
```

can optionally be deleted.
They are Apple metadata files and do not affect execution.

---

## 7. CPU Training Workaround

The original `train_worker` assumes GPU availability.
For CPU-only environments, replace it with a version that detects the device automatically.

Key idea:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)
```

Batches must also be moved:

```python
Batch(b[0].to(device), b[1].to(device), pad_idx)
```

CUDA-specific functions should only run if CUDA exists.


The complete function is given below:
```python
def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)

    if device.type == "cuda":
        torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model = model.to(device)

    module = model
    is_main_process = True

    if is_distributed and device.type == "cuda":
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    ).to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
    )

    train_state = TrainState()

    for epoch in range(config["num_epochs"]):

        model.train()
        print(f"[Epoch {epoch}] Training ====", flush=True)

        _, train_state = run_epoch(
            (Batch(b[0].to(device), b[1].to(device), pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        if device.type == "cuda":
            GPUtil.showUtilization()

        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)

        print(f"[Epoch {epoch}] Validation ====", flush=True)

        model.eval()
        sloss = run_epoch(
            (Batch(b[0].to(device), b[1].to(device), pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )

        print(sloss)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)
```

---

## 8. Model Loading Fix for CPU

If loading a model trained on GPU while running on CPU:

Replace:

```python
model.load_state_dict(torch.load("multi30k_model_final.pt"))
```

with:

```python
model.load_state_dict(torch.load("multi30k_model_final.pt", map_location="cpu"))
```

---

## 9. Performance Note

Running training on CPU is extremely slow.

Example observation:

* Even by the **second epoch**, runtime can exceed **20 minutes**.

GPU training is strongly recommended for practical use.

To make the model / training simpler for CPU (tho there is nothing such as free lunch!): we can make the following possible changes to the config:
```python
config = {
    "d_model": 512, # TRY: 128 ---- **NEW**
    "n_layers": 6, # TRY: 2 ---- **NEW**
    "batch_size": 32, # TRY: 16 (or even 8)
    "distributed": False,
    "num_epochs": 8, # TRY: 4
    "accum_iter": 10,
    "base_lr": 1.0,
    "max_padding": 72, # TRY: 40
    "warmup": 3000,
    "file_prefix": "multi30k_model_",
}
```

and yes, there are two unseen variables which the `train_worker` had defined inside, there we make the change from:

```python
d_model = 512
model = make_model(len(vocab_src), len(vocab_tgt), N=6)
```

to:

```python
d_model = config["d_model"] if "d_model" in config else 512
n_layers = config["n_layers"] if "n_layers" in config else 6
model = make_model(len(vocab_src), len(vocab_tgt), N=n_layers)
```

<text style="color: red; font-weight: bold;">
NOTE that the last 3 subsection on CPU efficiency is NOT implemented for distributed training (obviously) and is only for debugging / learning purposes. For distributed training, GPU is a must.
</text>
