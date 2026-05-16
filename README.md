# PMDM: A dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets

Official implementation of **PMDM**, a dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets, by Lei Huang.

## 📢 News

- Our paper is accepted by **Nature Communications** !! (https://doi.org/10.1038/s41467-024-46569-1)
- If you are interested in generating molecules from scatch (without protein pockets), please refer to our previous work [MDM](https://github.com/tencent-ailab/MDM)
- Please contact me if you are interested in my work and look for academic collaboration. (layne_huang@outlook.com).

[![biorxiv](https://img.shields.io/badge/biorxiv-526011-AE353A.svg)](https://www.biorxiv.org/content/10.1101/2023.01.28.526011v1.abstract)
[![DOI](https://zenodo.org/badge/753795380.svg)](https://zenodo.org/doi/10.5281/zenodo.10631313)

<div align="center">  
<img src="img/model.png" width="600">
</div>
<div align="center"> 
<img src="img/traj.gif" alt="GIF" width="400">
</div>

1. [Reporting Guide (汇报建议)](#reporting-guide-汇报建议)
2. [Dependencies](#dependencies)
   1. [Conda environment](#conda-environment)
   2. [QuickVina 2](#quickvina-2)
   3. [Pre-trained models](#pre-trained-models)
3. [Benchmarks](#benchmarks)
   1. [CrossDocked Benchmark](#crossdocked)
   2. [Binding MOAD](#binding-moad)
4. [Training](#training)
5. [Inference](#inference)
   1. [Test set sampling](#sample-molecules-for-all-pockets-in-the-test-set)
   2. [Sample molecules for a given pocket](#sample-molecules-for-a-given-pocket) 
   3. [Metrics](#metrics)
   4. [QuickVina2](#quickvina2)
6. [Citation](#citation)

## Reporting Guide (汇报建议)

> 请先明确你实际使用的配置文件（例如 `configs/crossdock_epoch.yml`），因为层数等超参均由配置读取。

### 总体流程（一行流程示意）
输入（蛋白口袋 + 配体特征）→ 扰动/扩散加噪 → 模型预测噪声/score → 反向采样生成分子

### 层数/卷积层/网络层（以 `configs/crossdock_epoch.yml` 为例）
- 全局 EGNN 层数：`num_convs=3`（global encoder）
- 局部 EGNN 层数：`num_convs_local=3`（local encoder）
- 蛋白/配体编码器交互层：`protein_num_convs=2`（`SchNetEncoder_protein`）
- 输出 MLP 头：每个 3 层（`grad_*_mlp`，input→hidden→hidden→output）
- 跨注意力块：1 个 `BasicTransformerBlock` 用于配体-蛋白交互

### β（beta schedule）
扩散过程的噪声调度由配置中的 `beta_schedule/beta_start/beta_end/num_diffusion_timesteps` 决定，并在 `MDM_full_pocket_coor_shared` 中生成 `betas` 用于每一步噪声强度。

### Embedding
- 时间步嵌入：`get_num_embedding` 生成正弦嵌入，经两层 MLP 投影后加到上下文（`temb.dense` + `temb_proj`）。
- 原子数嵌入（可选）：`atom_num_emb` 开关控制，流程与时间嵌入相同。

### 架构说明：无需 Decoder
这是扩散/score 模型，目标是从带噪输入预测噪声/梯度并进行反向采样，不需要自编码器式的 decoder；整体更像“条件编码器 + 噪声预测器”。

### Global (g) 与 Local (l) 的含义
g=global，l=local。两套边分别建模：
- local：短程/化学键邻域（`cutoff=3.0`）
- global：更长程口袋相互作用（`g_cutoff=6.0`）
两路输出在损失与采样时加权融合（例如 `pos_eq_global + pos_eq_local`，以及 `w_global_pos/w_local_pos`）。

### 模块—作用—关键超参—代码位置
| 模块 | 作用 | 关键超参（示例） | 代码位置 |
| --- | --- | --- | --- |
| 配置文件 | 统一管理层数与超参 | `num_convs/num_convs_local/protein_num_convs` | `configs/*.yml` |
| β 调度 | 控制扩散噪声强度 | `beta_schedule/beta_start/beta_end/num_diffusion_timesteps` | `models/epsnet/diffusion.py` |
| 时间/原子数嵌入 | 条件化扩散过程 | `time_emb/atom_num_emb` | `models/epsnet/diffusion.py` + `models/epsnet/MDM_pocket_coor_shared.py` |
| Global EGNN | 全局口袋建模 | `num_convs/g_cutoff` | `models/epsnet/MDM_pocket_coor_shared.py` |
| Local EGNN | 局部键/短程建模 | `num_convs_local/cutoff` | `models/epsnet/MDM_pocket_coor_shared.py` |
| 蛋白/配体编码器 | 抽取口袋与配体表示 | `protein_num_convs/encoder_cutoff` | `models/epsnet/MDM_pocket_coor_shared.py` + `models/encoders/schnet.py` |
| 跨注意力块 | 配体-蛋白交互 | `hidden_dim` | `models/epsnet/MDM_pocket_coor_shared.py` + `models/encoders/attention.py` |
| 输出 MLP 头 | 预测噪声/score | `mlp_act/hidden_dim` | `models/epsnet/MDM_pocket_coor_shared.py` |

### 汇报总结（3 点）
- 双扩散分支（global/local）同时建模全局口袋与局部化学键相互作用。
- 条件信息来自蛋白口袋，配体与口袋通过注意力交互融合。
- 通过扩散反向采样生成 3D 分子构象与原子特征。

## Dependencies

### Conda environment
Please use our environment file to install the environment.
```bash
# Clone the environment
conda env create -f mol.yml
# Activate the environment
conda activate mol
```
### QuickVina 2
For docking, install QuickVina 2:

```bash
wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1
chmod +x qvina2.1
```

Preparing the receptor for docking (pdb -> pdbqt) requires a new environment which is based on python 2x, so we need to create a new environment:
```bash
# Clone the environment
conda env create -f evaluation/env_adt.yml
# Activate the environment
conda activate adt
```
### Pre-trained models
The pre-trained models could be downloaded from [Zenodo](https://zenodo.org/records/10630921).

## Benchmarks
### CrossDocked

#### Data preparation
Download and extract the dataset is provided in [Zenodo](https://zenodo.org/records/10630921)

The original CrossDocked dataset can be found at https://bits.csb.pitt.edu/files/crossdock2020/

### Binding MOAD
#### Data preparation
Download the dataset
```bash
wget http://www.bindingmoad.org/files/biou/every_part_a.zip
wget http://www.bindingmoad.org/files/biou/every_part_b.zip
wget http://www.bindingmoad.org/files/csv/every.csv

unzip every_part_a.zip
unzip every_part_b.zip
```

## Training
We provide two training scripts **train.py** and **train_ddp_op.py** for single-GPU training and multi-GPU training.

Starting a new training run:
```bash
python -u train.py --config <config>.yml
```
The example configure file is in `configs/crossdock_epoch.yml`

Resuming a previous run:
```bash
python -u train.py --config <configure file path>
```
The config argument should be the upper path of the configure file.

## Inference
### Sample molecules for all pockets in the test set
```bash
python -u sample_batch.py --ckpt <checkpoint> --num_samples <number of samples> --sampling_type generalized
```

### Sample molecules for given customized pockets
```bash
python -u sample_for_pdb.py --ckpt <checkpoint> --pdb_path <pdb path> --num_atom <num atom> --num_samples <number of samples> --sampling_type generalized
```
`num_atom` is the number of atoms of generated molecules.

if you don't have the pocket pdb but the complex of protein and reference ligand, please run ```python split_pocket_ligand.py --path <pdb path> ```. It will automatically extract the pocket for you.

### Sample novel molecules given seed fragments
```bash
python -u sample_frag.py --ckpt <checkpoint> --pdb_path <pdb path> --mol_file <mole file> --keep_index <seed fragments index> --num_atom <num atom> --num_samples <number of samples> --sampling_type generalized
```
`num_atom` is the number of atoms of generated fragments. `keep_index` is the index of the atoms of the seed fragments.
You could utilize the following code to visualize the index of your molecule.
```
from rdkit import Chem
mol = Chem.SDMolSupplier(f)[0]
smiles = Chem.MolToSmiles(mol)
print(smiles)
mol.RemoveAllConformers()
for i, atom in enumerate(mol.GetAtoms()):
    atom.SetProp('molAtomMapNumber', str(i))
Draw.MolToImage(mol, size=(1000,1000))
```
For example, you could set keep index as 4 5 10 11 12 13 14 for the following molecule to generate novel molecules based on the desired fragment.

Here is an example command
```
python -u sample_frag.py --ckpt 500.pt --pdb_path data/2VUKcut10/2VUKcut10_pocket.pdb --mol_file data/2VUKcut10/2VUKcut10_ligand.sdf --keep_index 4 5 10 11 12 13 14 --num_atom 18 --num_samples 20 --sampling_type generalized
```
The reference generated molecule is shown as follows:
![sample_frag](https://github.com/Layne-Huang/PMDM/assets/34830172/5a4313b4-e2e9-4a70-95fa-dfaf054d3234)

### Sample novel molecules for linker 
```bash
python -u sample_linker.py --ckpt <checkpoint> --pdb_path <pdb path> --mol_file <mole file> --keep_index <seed fragments index> --num_atom <num atom> --num_samples <number of samples> --sampling_type generalized
```
`num_atom` is the number of atoms of generated fragments. `mask` is the index of the linker that you would like to replace in the original molecule.
For example, you could mask 6 7 8 9 10 11 to generate new linkers.

Here is an example command
```
python -u sample_linker.py --ckpt 500.pt --pdb_path data/3wzecut10/3wzecut10_pocket.pdb --mol_file data/3wzecut10/3wzecut10_ligand.sdf --mask 6 7 8 9 10 11 --num_atom 4 --num_samples 1 --sampling_type generalized --batch_size 1 -build_method reconstruct
```
The reference generated molecule is shown as follows:
![sample_linker](https://github.com/Layne-Huang/PMDM/assets/34830172/a4445170-e5a0-4403-adf8-0105990d66b4)







### Metrics
Evaluate the batch of generated molecules (You need to turn on the `save_results` arguments in sample* scripts)
```bash
python -u evaluate --path <molecule_path>
```

If you want to evaluate a single molecule, use `evaluate_single.py`.

### QuickVina2
First, convert all protein PDB files to PDBQT files using adt envrionment.
```bash
conda activate adt
prepare_receptor4.py -r {} -o {}
cd evaluation
```
Then, compute QuickVina scores:
```bash
conda deactivate
conda activate mol
python docking_2_single.py --receptor_file <prepapre_receptor4_outdir> --sdf_file <sdf file> --out_dir <qvina_outdir>
```
!!! You have to replace the path of your own mol and adt environment paths with the path in the scripts already.
### Citation
```
@article {Huang2023.01.28.526011,
	author = {Lei Huang and Tingyang Xu and Yang Yu and Peilin Zhao and Ka-Chun Wong and Hengtong Zhang},
	title = {A dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets},
	elocation-id = {2023.01.28.526011},
	year = {2023},
	doi = {10.1101/2023.01.28.526011},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/01/30/2023.01.28.526011},
	eprint = {https://www.biorxiv.org/content/early/2023/01/30/2023.01.28.526011.full.pdf},
	journal = {bioRxiv}
}
```

