rule all:
    input:
        "models/UNet_CKPT_best.pt"

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

