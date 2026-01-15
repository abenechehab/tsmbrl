#!/bin/bash
# Run experiments across multiple datasets and configurations

set -e

OUTPUT_DIR="results/raw"
mkdir -p "$OUTPUT_DIR"

# Datasets to evaluate
DATASETS=("door-human" "door-expert" "pen-human" "pen-expert")

# Models to test
MODELS=("chronos2")

# Prediction horizons
HORIZONS=(1 5 10)

# Lookback window
LOOKBACK=50

echo "========================================"
echo "TSMBRL Batch Experiments"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "Horizons: ${HORIZONS[*]}"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for horizon in "${HORIZONS[@]}"; do
            echo ""
            echo "Running: $dataset / $model / horizon=$horizon"
            echo "----------------------------------------"

            # With actions (if supported)
            output_file="${OUTPUT_DIR}/${dataset}_${model}_h${horizon}_with_actions.json"
            echo "  -> With actions: $output_file"

            python -m tsmbrl.inference \
                --dataset "$dataset" \
                --model "$model" \
                --lookback "$LOOKBACK" \
                --horizon "$horizon" \
                --with-actions \
                --max-episodes 50 \
                --output "$output_file" \
                2>&1 | tail -5

            # Without actions (baseline)
            output_file="${OUTPUT_DIR}/${dataset}_${model}_h${horizon}_no_actions.json"
            echo "  -> Without actions: $output_file"

            python -m tsmbrl.inference \
                --dataset "$dataset" \
                --model "$model" \
                --lookback "$LOOKBACK" \
                --horizon "$horizon" \
                --no-actions \
                --max-episodes 50 \
                --output "$output_file" \
                2>&1 | tail -5

        done
    done
done

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# Count results
n_results=$(find "$OUTPUT_DIR" -name "*.json" | wc -l)
echo "Total result files: $n_results"
