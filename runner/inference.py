# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import shutil
import subprocess
import time
import traceback
import urllib.request
from argparse import Namespace
from contextlib import nullcontext
from os.path import exists as opexists
from os.path import join as opjoin
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
from ml_collections.config_dict import ConfigDict

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config import parse_configs, parse_sys_args
from protenix.data.infer_data_pipeline import get_inference_dataloader
from protenix.model.HierAFold import HierAFold
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device
from protenix.web_service.dependency_url import URL
from runner.dumper import DataDumper
logger = logging.getLogger(__name__)
"""
Due to the fair-esm repository being archived, 
it can no longer be updated to support newer versions of PyTorch. 
Starting from PyTorch 2.6, the default value of the weights_only argument 
in torch.load has been changed from False to True, 
which enhances security but causes loading ESM models to fail 
with the following error:

_pickle.UnpicklingError: Weights only load failed. This file can still be loaded...
This error occurs because the model file contains argparse.Namespace, 
which is not allowed by default in the secure unpickling process of PyTorch 2.6+.

✅ Solution (Patch)
Since we cannot modify the fair-esm source code, 
we can apply a patch before calling load_model_and_alphabet_local 
by manually adding argparse.Namespace to PyTorch's safe globals list.
"""

torch.serialization.add_safe_globals([Namespace])


# --------------------------------------------------------------------------- #
# InterProScan / Nextflow domain-split helpers                                #
# --------------------------------------------------------------------------- #

def _is_nextflow_available() -> bool:
    """Return True if ``nextflow`` is found on PATH."""
    return shutil.which("nextflow") is not None


def _write_fasta(sequence: str, output_file: str, header: str = ">sequence") -> None:
    """Write a single protein sequence to a FASTA file (60-char line wrap)."""
    sequence = sequence.strip().upper()
    with open(output_file, "w") as fh:
        fh.write(header + "\n")
        for i in range(0, len(sequence), 60):
            fh.write(sequence[i : i + 60] + "\n")


def _run_interproscan(fasta_file: str, datadir: str) -> int:
    """
    Run InterProScan via the ``ebi-pf-team/interproscan6`` Nextflow pipeline
    (CATH-Gene3D application only) for a single FASTA file.

    Args:
        fasta_file: Path to the input FASTA file.
        datadir: Path to the InterProScan data directory.

    Returns:
        Return code of the subprocess (0 = success).
    """
    command = [
        "nextflow", "run", "ebi-pf-team/interproscan6",
        "-r", "6.0.0-beta",
        "-profile", "docker",
        "--datadir", datadir,
        "--input", fasta_file,
        "--applications", "CATH-Gene3D",
    ]
    logger.info(f"Running InterProScan: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"InterProScan finished for {fasta_file}")
        logger.debug(result.stdout)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"InterProScan failed for {fasta_file} (code {e.returncode}):\n{e.stderr}"
        )
        return e.returncode


def run_interproscan_for_input(
    input_json_path: str,
    gff3_output_dir: str,
    interproscan_datadir: str = "/data2/cxlu",
) -> Optional[str]:
    """
    Detect whether Nextflow is installed; if so, extract every protein-chain
    sequence from the HierAFold input JSON, write per-chain FASTA files,
    run the InterProScan Nextflow pipeline for each, and collect the resulting
    GFF3 files into ``gff3_output_dir``.

    The GFF3 files are named ``<sample_name>_seq_<chain_index>.fasta.gff3``
    so that :meth:`HierAFold.select_context_tokens` can locate them via
    ``parse_gff3_domains``.

    Args:
        input_json_path: Path to the HierAFold input JSON file.
        gff3_output_dir: Directory where GFF3 annotation files will be
            collected.  Created automatically if it does not exist.
        interproscan_datadir: Path to the InterProScan reference data
            directory used by the Nextflow pipeline.

    Returns:
        Path to ``gff3_output_dir`` if Nextflow is available and at least one
        GFF3 was produced; ``None`` otherwise (domain splitting will fall back
        to PAE-based detection inside the model).
    """
    if not _is_nextflow_available():
        logger.info(
            "Nextflow not found on PATH — skipping GFF3 domain annotation. "
            "Domain boundaries will be estimated from PAE maps during inference."
        )
        return None

    logger.info("Nextflow detected — running InterProScan domain annotation.")

    with open(input_json_path, "r") as fh:
        entries = json.load(fh)

    Path(gff3_output_dir).mkdir(parents=True, exist_ok=True)
    fasta_tmp_dir = opjoin(gff3_output_dir, "_fasta_tmp")
    Path(fasta_tmp_dir).mkdir(parents=True, exist_ok=True)

    any_produced = False

    for entry in entries:
        sample_name = entry.get("name", "sample")
        sequences = entry.get("sequences", [])

        for seq_idx, seq_entry in enumerate(sequences):
            if "proteinChain" not in seq_entry:
                continue  # skip nucleic acids and ligands
            sequence = seq_entry["proteinChain"].get("sequence", "")
            if not sequence:
                continue

            fasta_filename = f"{sample_name}_seq_{seq_idx}.fasta"
            fasta_path = opjoin(fasta_tmp_dir, fasta_filename)
            _write_fasta(sequence, fasta_path, header=f">{sample_name}_chain_{seq_idx}")

            logger.info(
                f"Running InterProScan for {sample_name} chain {seq_idx} "
                f"(length {len(sequence)})"
            )
            ret = _run_interproscan(fasta_path, interproscan_datadir)

            if ret == 0:
                # The Nextflow pipeline writes results alongside the FASTA file;
                # collect any produced GFF3 into gff3_output_dir.
                expected_gff3 = fasta_path + ".gff3"
                dest_gff3 = opjoin(gff3_output_dir, fasta_filename + ".gff3")
                if opexists(expected_gff3):
                    shutil.copy2(expected_gff3, dest_gff3)
                    logger.info(f"GFF3 saved: {dest_gff3}")
                    any_produced = True
                else:
                    logger.warning(
                        f"InterProScan succeeded but GFF3 not found at {expected_gff3}"
                    )

    if any_produced:
        return gff3_output_dir
    else:
        logger.info(
            "InterProScan produced no GFF3 files — "
            "domain boundaries will be estimated from PAE maps."
        )
        return None


class InferenceRunner(object):
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")
        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set `CUTLASS_PATH` environment variable according to the instructions at https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            if env is not None:
                logging.info(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        self.model = HierAFold(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        checkpoint_path = (
            f"{self.configs.load_checkpoint_dir}/{self.configs.model_name}.pt"
        )
        print('checkpoint path:', checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = next(iter(checkpoint["model"]))
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module."):]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=self.configs.load_strict,
        )
        self.model.eval()
        self.print(f"Finish loading checkpoint.")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ):
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    # Adapted from runner.train.Trainer.evaluate
    @torch.no_grad()
    def predict(
        self,
        data: Mapping[str, Mapping[str, Any]],
        gff3_dir: Optional[str] = None,
    ) -> dict:
        """
        Run a single forward pass of HierAFold.

        Args:
            data: Batch dict produced by the inference dataloader.
            gff3_dir: Optional path to GFF3 domain annotation files produced
                by :func:`run_interproscan_for_input`.  When ``None``, domain
                boundaries are estimated from PAE maps inside the model.

        Returns:
            Prediction dictionary from :meth:`HierAFold.forward`.
        """
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)
        sample_name = data["sample_name"]

        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                # gff3_dir=gff3_dir,
                gff3_dir='/home/u3590540/cxlu/protein/Protenix4/Protenix/Interpro_subunit',
                pdb_id=sample_name,
            )

        return prediction

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        self.model.configs = new_configs


def download_infercence_cache(configs: Any) -> None:
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_length = 30
        filled_length = int(bar_length * percent // 100)
        bar = "=" * filled_length + "-" * (bar_length - filled_length)

        status = f"\r[{bar}] {percent:.1f}%"
        print(status, end="", flush=True)

        if downloaded >= total_size:
            print()

    def download_from_url(tos_url, checkpoint_path, check_weight=True):
        urllib.request.urlretrieve(
            tos_url, checkpoint_path, reporthook=progress_callback
        )
        if check_weight:
            try:
                ckpt = torch.load(checkpoint_path)
                del ckpt
            except:
                os.remove(checkpoint_path)
                raise RuntimeError(
                    "Download model checkpoint failed, please download by yourself with "
                    f"wget {tos_url} -O {checkpoint_path}"
                )

    for cache_name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
    ):
        cur_cache_fpath = configs["data"][cache_name]
        if not opexists(cur_cache_fpath):
            os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
            tos_url = URL[cache_name]
            assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                f"{cache_name} file name is incorrect, `{tos_url}` and "
                f"`{cur_cache_fpath}`. Please check and try again."
            )
            logger.info(
                f"Downloading data cache from\n {tos_url}... to {cur_cache_fpath}"
            )
            download_from_url(tos_url, cur_cache_fpath, check_weight=False)

    checkpoint_path = f"{configs.load_checkpoint_dir}/{configs.model_name}.pt"
    checkpoint_dir = configs.load_checkpoint_dir

    if not opexists(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        tos_url = URL[configs.model_name]
        logger.info(
            f"Downloading model checkpoint from\n {tos_url}... to {checkpoint_path}"
        )
        download_from_url(tos_url, checkpoint_path)

    if "esm" in configs.model_name:  # currently esm only support 3b model
        esm_3b_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D.pt"
        if not opexists(esm_3b_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}... to {esm_3b_ckpt_path}"
            )
            download_from_url(tos_url, esm_3b_ckpt_path)
        esm_3b_ckpt_path2 = f"{checkpoint_dir}/esm2_t36_3B_UR50D-contact-regression.pt"
        if not opexists(esm_3b_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D-contact-regression"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}... to {esm_3b_ckpt_path2}"
            )
            download_from_url(tos_url, esm_3b_ckpt_path2)
    if "ism" in configs.model_name:
        esm_3b_ism_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism.pt"

        if not opexists(esm_3b_ism_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D_ism"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}... to {esm_3b_ism_ckpt_path}"
            )
            download_from_url(tos_url, esm_3b_ism_ckpt_path)

        esm_3b_ism_ckpt_path2 = f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism-contact-regression.pt"  # the same as esm_3b_ckpt_path2
        if not opexists(esm_3b_ism_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D_ism-contact-regression"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}... to {esm_3b_ism_ckpt_path2}"
            )
            download_from_url(tos_url, esm_3b_ism_ckpt_path2)


def update_inference_configs(configs: Any, N_token: int):
    # Setting the default inference configs for different N_token and N_atom
    # when N_token is larger than 3000, the default config might OOM even on a
    # A100 80G GPUS,
    if N_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif N_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True

    return configs


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    # Data
    logger.info(f"Loading data from\n{configs.input_json_path}")
    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as e:
        error_message = f"{e}:\n{traceback.format_exc()}"
        logger.info(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a") as f:
            f.write(error_message)
        return

    # Run InterProScan domain annotation once for all sequences in the input
    # JSON before entering the inference loop.  If Nextflow is not installed
    # (or produces no output), gff3_dir is None and the model falls back to
    # PAE-based domain detection automatically.
    gff3_dir = run_interproscan_for_input(
        input_json_path=configs.input_json_path,
        gff3_output_dir=opjoin(configs.dump_dir, "gff3_domains"),
        interproscan_datadir=getattr(configs, "interproscan_datadir", "/data2/cxlu"),
    )
    if gff3_dir is not None:
        logger.info(f"GFF3 domain annotations available at: {gff3_dir}")
    else:
        logger.info(
            "No GFF3 domain annotations — domain boundaries will be "
            "inferred from PAE maps at runtime."
        )

    num_data = len(dataloader.dataset)
    t0_start = time.time()
    for seed in configs.seeds:
        seed_everything(seed=seed, deterministic=configs.deterministic)
        t1_start = time.time()
        for batch in dataloader:
            try:
                t2_start = time.time()
                data, atom_array, data_error_message = batch[0]
                sample_name = data["sample_name"]

                if len(data_error_message) > 0:
                    logger.info(data_error_message)
                    with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a") as f:
                        f.write(data_error_message)
                    continue

                logger.info(
                    (
                        f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                        f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                        f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                    )
                )
                new_configs = update_inference_configs(configs, data["N_token"].item())
                runner.update_model_configs(new_configs)
                prediction = runner.predict(data, gff3_dir=gff3_dir)
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type=data["entity_poly_type"],
                )
                t2_end = time.time()
                logger.info(
                    f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} succeeded. Model forward time: {t2_end-t2_start}s.\n"
                    f"Results saved to {configs.dump_dir}"
                )
                torch.cuda.empty_cache()
            except Exception as e:
                error_message = f"[Rank {DIST_WRAPPER.rank}]{data['sample_name']} {e}:\n{traceback.format_exc()}"
                logger.info(error_message)
                # Save error info
                with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a") as f:
                    f.write(error_message)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
        t1_end = time.time()
        logger.info(
            f"[Rank {DIST_WRAPPER.rank}] seed {seed} succeeded. Total task time: {t1_end-t1_start}s.\n"
        )
    t0_end = time.time()
    logger.info(
        f"[Rank {DIST_WRAPPER.rank}] job succeeded. Total job time: {t0_end-t0_start}s.\n"
    )


def main(configs: Any) -> None:
    # Runner
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)


def update_gpu_compatible_configs(configs: Any) -> None:
    def is_gpu_capability_between_7_and_8():
        # 7.0 <= device_capability < 8.0
        if not torch.cuda.is_available():
            return False

        capability = torch.cuda.get_device_capability()
        major, minor = capability
        cc = major + minor / 10.0
        if 7.0 <= cc < 8.0:
            return True
        return False

    if is_gpu_capability_between_7_and_8():
        # Some kernels and BF16 aren’t supported on V100 — enforce specific configurations to work around it.
        configs.dtype = "fp32"
        configs.triangle_attention = "torch"
        configs.triangle_multiplicative = "torch"
        logger.info(
            "GPU capability is between 7.0 and 8.0, enforce fp32 and torch kernels for triangle attention and multiplicative."
        )
    return configs


def run() -> None:
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )
    model_name = configs.model_name
    _, model_size, model_feature, model_version = model_name.split("_")
    logger.info(
        f"Inference by Protenix: model_size: {model_size}, with_feature: {model_feature.replace('-',', ')}, model_version: {model_version}, dtype: {configs.dtype}"
    )
    model_specfics_configs = ConfigDict(model_configs[model_name])
    # update model specific configs
    configs.update(model_specfics_configs)
    configs = update_gpu_compatible_configs(configs)
    logger.info(
        f"Triangle_multiplicative kernel: {configs.triangle_multiplicative}, Triangle_attention kernel: {configs.triangle_attention}"
    )
    logger.info(
        f"enable_diffusion_shared_vars_cache: {configs.enable_diffusion_shared_vars_cache}, enable_efficient_fusion: {configs.enable_efficient_fusion}, enable_tf32: {configs.enable_tf32}"
    )
    download_infercence_cache(configs)
    main(configs)


if __name__ == "__main__":
    run()
