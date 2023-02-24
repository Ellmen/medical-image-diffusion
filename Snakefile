rule all:
    input:
        "outputs/model_stats.csv",
        "outputs/dag.pdf"

rule render_dag:
    input:
        "Snakefile"
    output:
        "outputs/dag.pdf"
    shell:
        "snakemake --dag | dot -Tpdf > outputs/dag.pdf"

rule train_segmentation:
    input:
        "data/DRIVE/"
    output:
        "models/UNet_CKPT_best.pt"
    script:
        "scripts/train_segmentation.py"

rule train_segmentation_high_epochs:
    input:
        "data/DRIVE/"
    output:
        "models/UNet_high_epochs_CKPT_best.pt"
    script:
        "scripts/train_segmentation_high_epochs.py"

rule train_segmentation_high_momentum:
    input:
        "data/DRIVE/"
    output:
        "models/UNet_high_momentum_CKPT_best.pt"
    script:
        "scripts/train_segmentation_high_momentum.py"

rule test_models:
    input:
        "data/DRIVE/",
        "models/UNet_CKPT_best.pt",
        "models/UNet_high_epochs_CKPT_best.pt",
        "models/UNet_high_momentum_CKPT_best.pt"
    output:
        "outputs/model_stats.csv"
    script:
        "scripts/test_models.py"
