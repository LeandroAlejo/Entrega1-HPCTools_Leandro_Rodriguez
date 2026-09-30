#!/bin/bash
#SBATCH --job-name=bert_squad_profiler
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=bert_profiler_results_%j.log

# 1. Limpiar y cargar módulos del CESGA
module --force purge
module load cesga/2020 python/3.10.8

# 2. Activar exclusivamente el entorno local de la tarea
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$SCRIPT_DIR/entornoTarea1"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: No se encontró el entorno virtual en $VENV_DIR."
    echo "Es necesario ejecutar primero: bash setup_env.sh"
    exit 1
fi

TMP_DIR=$(mktemp -d)

# 500 pasos evaluados (más 10 de warmup) para superar holgadamente el minuto de entrenamiento
STEPS=500
WARMUP=10

echo "=========================================================="
echo " BERT SQUAD BENCHMARKS WITH PROFILER (500 STEPS) "
echo "=========================================================="

echo -e "\n[1/5] RUNNING BASELINE..."
python3 "$SCRIPT_DIR/train_baseline.py" \
    --batch_size 16 \
    --num_workers 0 \
    --steps $STEPS \
    --warmup $WARMUP \
    --profile | tee "$TMP_DIR/run1.txt"

echo -e "\n[2/5] RUNNING OPTIMIZED DATALOADER..."
python3 "$SCRIPT_DIR/train_baseline.py" \
    --batch_size 16 \
    --num_workers 16 \
    --pin_memory \
    --steps $STEPS \
    --warmup $WARMUP \
    --profile | tee "$TMP_DIR/run2.txt"

echo -e "\n[3/5] RUNNING MIXED PRECISION (BF16)..."
python3 "$SCRIPT_DIR/train_baseline.py" \
    --batch_size 16 \
    --num_workers 16 \
    --pin_memory \
    --use_amp \
    --steps $STEPS \
    --warmup $WARMUP \
    --profile | tee "$TMP_DIR/run3.txt"

echo -e "\n[4/5] RUNNING DOUBLE BATCH SIZE..."
python3 "$SCRIPT_DIR/train_baseline.py" \
    --batch_size 32 \
    --num_workers 16 \
    --pin_memory \
    --use_amp \
    --steps $STEPS \
    --warmup $WARMUP \
    --profile | tee "$TMP_DIR/run4.txt"

echo -e "\n[5/5] RUNNING COMPILED MODEL..."
python3 "$SCRIPT_DIR/train_baseline.py" \
    --batch_size 32 \
    --num_workers 16 \
    --pin_memory \
    --use_amp \
    --compile \
    --steps $STEPS \
    --warmup $WARMUP \
    --profile | tee "$TMP_DIR/run5.txt"

# ==============================================================================
# 3. EXTRACCIÓN Y GENERACIÓN DE TABLAS COMPARATIVAS FINALES
# ==============================================================================

extract_metrics() {
    local file=$1
    local t_sec=$(grep -m 1 "training time medido" "$file" | awk -F: '{print $2}' | awk '{print $1}')
    local sps=$(grep -m 1 "samples / s" "$file" | awk -F: '{print $2}' | xargs)
    local tflops=$(grep -m 1 "achieved TFLOP/s" "$file" | awk -F: '{print $2}' | xargs)
    local mfu_real=$(grep -m 1 "MFU % (Hardware real)" "$file" | awk -F: '{print $2}' | awk '{print $1}')
    local mfu_global=$(grep -m 1 "MFU % (vs Tensor Cores)" "$file" | awk -F: '{print $2}' | awk '{print $1}')
    local mem=$(grep -m 1 "peak GPU memory" "$file" | awk -F: '{print $2}' | awk '{print $1}')
    local loss=$(grep -m 1 "final loss" "$file" | awk -F: '{print $2}' | xargs)

    echo "$t_sec|$sps|$tflops|$mfu_real|$mfu_global|$mem|$loss"
}

M1=$(extract_metrics "$TMP_DIR/run1.txt")
M2=$(extract_metrics "$TMP_DIR/run2.txt")
M3=$(extract_metrics "$TMP_DIR/run3.txt")
M4=$(extract_metrics "$TMP_DIR/run4.txt")
M5=$(extract_metrics "$TMP_DIR/run5.txt")

cat << 'EOF'

=============================================================================================================================================
TABLA 1: RESUMEN COMPARATIVO DE RENDIMIENTO (NVIDIA A100-PCIE-40GB)
=============================================================================================================================================
Etapa / Configuración              Precisión   BS   Workers  Throughput   Achieved     MFU Real      MFU Global    Peak VRAM   Speedup
                                                             (samples/s)  (TFLOP/s)   (Hardware)    (vs 312 TF)      (MB)
---------------------------------------------------------------------------------------------------------------------------------------------
EOF

awk -v m1="$M1" -v m2="$M2" -v m3="$M3" -v m4="$M4" -v m5="$M5" 'BEGIN {
    split(m1, a1, "|"); split(m2, a2, "|"); split(m3, a3, "|"); split(m4, a4, "|"); split(m5, a5, "|");
    base = a1[2] > 0 ? a1[2] : 1;
    printf "[1/5] Baseline                     FP32       16      0   %10.2f   %10.4f   %10s*  %10s    %8.2f MB    1.00x\n", a1[2], a1[3], a1[4], a1[5], a1[6];
    printf "[2/5] Optimized DataLoader         FP32       16     16   %10.2f   %10.4f   %10s*  %10s    %8.2f MB   %5.2fx\n", a2[2], a2[3], a2[4], a2[5], a2[6], a2[2]/base;
    printf "[3/5] Mixed Precision (AMP)        BF16       16     16   %10.2f   %10.4f   %10s   %10s    %8.2f MB   %5.2fx\n", a3[2], a3[3], a3[4], a3[5], a3[6], a3[2]/base;
    printf "[4/5] Double Batch Size            BF16       32     16   %10.2f   %10.4f   %10s   %10s    %8.2f MB   %5.2fx\n", a4[2], a4[3], a4[4], a4[5], a4[6], a4[2]/base;
    printf "[5/5] Compiled Model (Inductor)    BF16       32     16   %10.2f   %10.4f   %10s   %10s    %8.2f MB   %5.2fx\n", a5[2], a5[3], a5[4], a5[5], a5[6], a5[2]/base;
}'

cat << 'EOF'
---------------------------------------------------------------------------------------------------------------------------------------------
* En FP32 (Etapas 1 y 2), el hardware opera sobre CUDA Cores convencionales (techo de 19.5 TFLOP/s).
  En BF16 (Etapas 3, 4 y 5), se activan los Tensor Cores de la arquitectura Ampere (techo de 312.0 TFLOP/s).
=============================================================================================================================================

=============================================================================================================================================
TABLA 2: CUMPLIMIENTO DE MÉTRICAS EXIGIDAS EN LA MEMORIA DEL LABORATORIO
=============================================================================================================================================
Etapa                              samples / s     peak GPU memory     achieved TFLOP/s        MFU %           final loss (500 steps)
---------------------------------------------------------------------------------------------------------------------------------------------
EOF

awk -v m1="$M1" -v m2="$M2" -v m3="$M3" -v m4="$M4" -v m5="$M5" 'BEGIN {
    split(m1, a1, "|"); split(m2, a2, "|"); split(m3, a3, "|"); split(m4, a4, "|"); split(m5, a5, "|");
    printf "[1/5] Baseline                %10.2f         %8.2f MB         %10.4f          %10s             %10s\n", a1[2], a1[6], a1[3], a1[5], a1[7];
    printf "[2/5] Optimized DataLoader    %10.2f         %8.2f MB         %10.4f          %10s             %10s\n", a2[2], a2[6], a2[3], a2[5], a2[7];
    printf "[3/5] Mixed Precision (BF16)  %10.2f         %8.2f MB         %10.4f          %10s             %10s\n", a3[2], a3[6], a3[3], a3[5], a3[7];
    printf "[4/5] Double Batch Size       %10.2f         %8.2f MB         %10.4f          %10s             %10s\n", a4[2], a4[6], a4[3], a4[5], a4[7];
    printf "[5/5] Compiled Model          %10.2f         %8.2f MB         %10.4f          %10s             %10s\n", a5[2], a5[6], a5[3], a5[5], a5[7];
}'

cat << 'EOF'

=============================================================================================================================================
TABLA 3: TIEMPOS DE ENTRENAMIENTO MEDIDOS (1 GPU NVIDIA A100)
=============================================================================================================================================
Etapa                              Pasos Medidos   Batch Size   Throughput       Tiempo Medido       Tiempo Medido
                                                                (samples/s)        (Segundos)          (Minutos)
---------------------------------------------------------------------------------------------------------------------------------------------
EOF

awk -v m1="$M1" -v m2="$M2" -v m3="$M3" -v m4="$M4" -v m5="$M5" -v st="$STEPS" 'BEGIN {
    split(m1, a1, "|"); split(m2, a2, "|"); split(m3, a3, "|"); split(m4, a4, "|"); split(m5, a5, "|");

    t1 = a1[1]; t2 = a2[1]; t3 = a3[1]; t4 = a4[1]; t5 = a5[1];

    printf "[1/5] Baseline                       %3d            16     %10.2f        %8.2f s          %5.2f min\n", st, a1[2], t1, t1/60;
    printf "[2/5] Optimized DataLoader           %3d            16     %10.2f        %8.2f s          %5.2f min\n", st, a2[2], t2, t2/60;
    printf "[3/5] Mixed Precision (BF16)         %3d            16     %10.2f        %8.2f s          %5.2f min\n", st, a3[2], t3, t3/60;
    printf "[4/5] Double Batch Size              %3d            32     %10.2f        %8.2f s          %5.2f min\n", st, a4[2], t4, t4/60;
    printf "[5/5] Compiled Model                 %3d            32     %10.2f        %8.2f s          %5.2f min\n", st, a5[2], t5, t5/60;
}'

cat << 'EOF'
=============================================================================================================================================
EOF

rm -rf "$TMP_DIR"
