import argparse
import time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator

# ==============================================================================
# 1. PARÁMETROS DE EJECUCIÓN (LÍNEA DE COMANDOS)
# Se definen flags independientes para probar metódicamente cada optimización:
# fijar memoria, número de workers en CPU, precisión mixta y compilación de grafo.
# ==============================================================================
parser = argparse.ArgumentParser(description="PyTorch Native BERT SQuAD Profiler")
parser.add_argument('--batch_size', type=int, default=16, 
                    help="Número de secuencias procesadas simultáneamente. Controla el paralelismo en la GPU.")
parser.add_argument('--num_workers', type=int, default=0, 
                    help="Procesos paralelos de CPU para pre-cargar batches. Si es 0, el proceso principal bloquea la GPU.")
parser.add_argument('--pin_memory', action='store_true', 
                    help="Asigna páginas bloqueadas en RAM (pinned host memory) para permitir transferencias PCIe asíncronas.")
parser.add_argument('--use_amp', action='store_true', 
                    help="Habilita torch.autocast en bfloat16 para derivar GEMMs densas a los Tensor Cores de la A100.")
parser.add_argument('--compile', action='store_true', 
                    help="Ejecuta torch.compile para capturar el grafo y fusionar kernels elementwise contiguos.")
parser.add_argument('--profile', action='store_true', 
                    help="Activa torch.profiler para desglosar el tiempo consumido por cada kernel de CUDA y la CPU.")
parser.add_argument('--steps', type=int, default=100, 
                    help="Iteraciones de entrenamiento medidas en estado estacionario.")
parser.add_argument('--warmup', type=int, default=10, 
                    help="Pasos previos descartados para evitar penalizaciones por compilación JIT y calentamiento de caché.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. CARGA DE MODELO Y TOKENIZADOR
# Usamos BERT-base para Question Answering (SQuAD).
# Los avisos UNEXPECTED (pesos de MLM/NSP del pre-entrenamiento descartados) y
# MISSING (la nueva capa 'qa_outputs' inicializada aleatoriamente) son esperados.
# ==============================================================================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)

# ==============================================================================
# 3. CÁLCULO DINÁMICO DE FLOPs POR SECUENCIA
# FlopCounterMode rastrea a nivel de operador C++ (ATen) la cantidad exacta de
# operaciones de coma flotante que demanda un forward + backward completo para
# una longitud de 384 tokens, evitando aproximaciones manuales erróneas.
# ==============================================================================
def measure_model_flops(test_model):
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_LENGTH), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((1, MAX_LENGTH), dtype=torch.long, device=device)
    dummy_start_positions = torch.tensor([0], dtype=torch.long, device=device)
    dummy_end_positions = torch.tensor([0], dtype=torch.long, device=device)
    
    dummy_batch = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask,
        "start_positions": dummy_start_positions,
        "end_positions": dummy_end_positions
    }

    # Conteo forward
    with FlopCounterMode(display=False) as fwd_counter:
        _ = test_model(**dummy_batch)
    forward_flops = fwd_counter.get_total_flops()

    # Conteo ciclo completo (forward + backward)
    with FlopCounterMode(display=False) as step_counter:
        out = test_model(**dummy_batch)
        loss = out.loss
        loss.backward()
    train_step_flops = step_counter.get_total_flops()

    # Limpieza: dejamos el modelo y la GPU limpios de gradientes y tensores temporales
    test_model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    macs_forward = forward_flops // 2
    tflops_per_seq = train_step_flops / 1e12

    return macs_forward, forward_flops, train_step_flops, tflops_per_seq

macs_fwd, fwd_flops, total_flops_per_seq, tflops_per_sequence = measure_model_flops(model)

print("\n" + "="*60)
print(" CÓMPUTO CALCULADO AUTOMÁTICAMENTE (POR SECUENCIA)")
print(f" MACs Forward (1 seq)       : {macs_fwd:,}")
print(f" FLOPs Forward (1 seq)      : {fwd_flops:,}")
print(f" FLOPs Train (Fwd + Bwd)    : {total_flops_per_seq:,}")
print(f" Constante por secuencia    : {tflops_per_sequence:.8e} TFLOPs")
print("="*60)

# ==============================================================================
# 4. PREPARACIÓN DEL DATASET Y PIPELINE DE ENTRADA
# El uso de drop_last=True es crucial al compilar con PyTorch: garantiza que
# todas las matrices conserven una dimensión fija constante (evitando que un
# batch final más pequeño dispare costosas recompilaciones de kernels).
# ==============================================================================
raw_datasets = load_dataset("rajpurkar/squad", split="train")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=MAX_LENGTH, truncation="only_second", stride=DOC_STRIDE,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions, end_positions = [], []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1: idx += 1
        context_start = idx
        while sequence_ids[idx] == 1: idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_idx = context_start
            while token_idx <= context_end and offset[token_idx][0] <= start_char: token_idx += 1
            start_positions.append(token_idx - 1)
            
            token_idx = context_end
            while token_idx >= context_start and offset[token_idx][1] >= end_char: token_idx -= 1
            end_positions.append(token_idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

train_dataset = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets.column_names)
train_dataset.set_format("torch")

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    collate_fn=default_data_collator,
    drop_last=True
)

# torch.compile: analiza el flujo de ejecución mediante TorchDynamo y genera
# kernels de Triton mediante TorchInductor fusionando capas de activación y normalización
if args.compile:
    model = torch.compile(model)

optimizer = AdamW(model.parameters(), lr=3e-5)
model.train()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

step = 0
start_time = None
total_samples = 0
last_loss_tensor = None

# ==============================================================================
# 5. FUNCIÓN DE PASO DE ENTRENAMIENTO ASÍNCRONO
# Principio de asincronía en PyTorch:
# - non_blocking=True solapa la transferencia de host a device mientras la GPU calcula.
# - zero_grad(set_to_none=True) libera memoria en lugar de escribir ceros explícitos.
# - Devolvemos el tensor 'loss' sin llamar a .item() para no forzar la sincronización
#   del hilo de Python con la GPU en cada iteración del bucle.
# ==============================================================================
def train_one_step(batch):
    batch = {k: v.to(device, non_blocking=args.pin_memory) for k, v in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    
    if args.use_amp:
        # Autocast bfloat16: mantiene el rango dinámico de FP32 y usa Tensor Cores
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(**batch)
            loss = output.loss
    else:
        # FP32 puro: cálculos canalizados a CUDA Cores estándar
        output = model(**batch)
        loss = output.loss
        
    loss.backward()
    optimizer.step()
    return loss

print(f"\n--- BS={args.batch_size} | Workers={args.num_workers} | PinMem={args.pin_memory} | AMP={args.use_amp} | Compile={args.compile} | Profile={args.profile} ---")

# ==============================================================================
# 6. BUCLE DE ENTRENAMIENTO Y TRAZADO CON PROFILER
# - schedule(wait=1, warmup=1, active=3): limita la captura a ventanas pequeñas
#   para evitar agotar la memoria RAM del sistema con eventos de rastreo.
# - time.perf_counter() con torch.cuda.synchronize(): medición precisa del hardware.
# ==============================================================================
if args.profile:
    from torch.profiler import profile, schedule, ProfilerActivity
    
    prof_schedule = schedule(wait=1, warmup=1, active=3)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True
    ) as prof:
        for batch in train_loader:
            if step == args.warmup:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                total_samples = 0

            last_loss_tensor = train_one_step(batch)
            prof.step()
            step += 1

            if step > args.warmup:
                total_samples += args.batch_size

            if step >= (args.warmup + args.steps):
                break

    print("\n=== PROFILER SUMMARY ===")
    sort_metric = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_metric, row_limit=10))

else:
    for batch in train_loader:
        if step == args.warmup:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            total_samples = 0

        last_loss_tensor = train_one_step(batch)
        step += 1

        if step > args.warmup:
            total_samples += args.batch_size

        if step >= (args.warmup + args.steps):
            break

# ==============================================================================
# 7. MÉTRICAS FINALES Y DISTINCIÓN DE MFU SEGÚN PRECISIÓN (FP32 vs BF16)
# ==============================================================================
if torch.cuda.is_available():
    torch.cuda.synchronize()  # Aseguramos que la GPU termine antes de detener el temporizador
    elapsed = time.perf_counter() - start_time if start_time else 1.0
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
else:
    elapsed = time.perf_counter() - start_time if start_time else 1.0
    peak_mem_mb = 0.0

# Obtenemos el valor escalar únicamente al terminar el proceso de medición
final_loss = last_loss_tensor.item() if last_loss_tensor is not None else 0.0
samples_per_sec = total_samples / elapsed
achieved_tflops = samples_per_sec * tflops_per_sequence

# Selección del techo teórico según la precisión activa:
# FP32 estándar: 19.5 TFLOP/s | BF16 (Tensor Cores): 312.0 TFLOP/s
hardware_peak_tflops = 312.0 if args.use_amp else 19.5
real_hardware_mfu = (achieved_tflops / hardware_peak_tflops) * 100
normalized_tensor_core_mfu = (achieved_tflops / 312.0) * 100

print(f"\nResultados tras {args.steps} pasos (descartando {args.warmup} de warmup):")
print(f" training time medido      : {elapsed:.2f} s ({elapsed/60:.2f} min)")
print(f" samples / s               : {samples_per_sec:.2f}")
print(f" peak GPU memory           : {peak_mem_mb:.2f} MB")
print(f" achieved TFLOP/s          : {achieved_tflops:.4f}")
print(f" MFU % (Hardware real)     : {real_hardware_mfu:.2f}% (Techo: {hardware_peak_tflops} TFLOP/s)")
print(f" MFU % (vs Tensor Cores)   : {normalized_tensor_core_mfu:.2f}% (Techo: 312.0 TFLOP/s)")
print(f" final loss                : {final_loss:.6f}")