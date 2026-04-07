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
import logging
import os
import time
import traceback
import urllib.request
from argparse import Namespace
from contextlib import nullcontext
from os.path import exists as opexists
from os.path import join as opjoin
from typing import Any, Mapping

import torch
import torch.distributed as dist
from ml_collections.config_dict import ConfigDict

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config import parse_configs, parse_sys_args
from protenix.data.infer_data_pipeline import get_inference_dataloader
from protenix.model.protenix import Protenix
from protenix.model.HiraryFold import ProtenixSelect
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device
from protenix.web_service.dependency_url import URL
from runner.dumper import DataDumper
from protenix.data.filter import Filter
from protenix.data.parser import AddAtomArrayAnnot, MMCIFParser
from typing import Any, Mapping, Tuple
from pathlib import Path
from protenix.data.data_pipeline import DataPipeline
from typing import Optional
from biotite.structure import AtomArray, get_chain_starts, get_residue_starts
import numpy as np
# from runner.supplment_CA import generate_symmetric
from protenix.data.constants import STD_RESIDUES
from protenix.data.utils import get_lig_lig_bonds, get_ligand_polymer_bond_mask
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
def gen_a_bioassembly_data_dict(
    mmcif: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
) -> Optional[dict]:
    """
    Generates bioassembly data from an mmCIF file and saves it to the specified output directory.

    Args:
        mmcif (Path): Path to the mmCIF file.
        bioassembly_output_dir (Path): Directory where the bioassembly data will be saved.
        cluster_file (Optional[Path]): Path to the cluster file, if available.
        distillation (bool, optional): Flag indicating whether to use the 'Distillation' setting. Defaults to False.

    Returns:
        Optional[list[dict]]: A list of sample indices if data is successfully generated, otherwise None.
    """
    if distillation:
        dataset = "Distillation"
    else:
        dataset = "WeightedPDB"

    sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
        mmcif, cluster_file, dataset
    )

    return bioassembly_dict

def atom_array_to_input_json(
    atom_array: AtomArray,
    parser: MMCIFParser,
    assembly_id: str = None,
    output_json: str = None,
    sample_name=None,
    save_entity_and_asym_id=False,
) -> Tuple[list, list]:
    """
    Convert a Biotite AtomArray to a dict that can be used as input to the model.

    Args:
        atom_array (AtomArray): Biotite Atom array.
        parser (MMCIFParser): Instantiated Protenix MMCIFParer.
        assembly_id (str, optional): Assembly ID. Defaults to None.
        output_json (str, optional): Output json file path. Defaults to None.
        sample_name (_type_, optional): The "name" filed in json file. Defaults to None.
        save_entity_and_asym_id (bool, optional): Whether to save entity and asym ids to json.
                                                  Defaults to False.

    Returns:
        dict: Protenix input json dict.
    """
    # get sequences after modified AtomArray
    entity_seq = parser.get_sequences(atom_array)

    # add unique chain id
    atom_array = AddAtomArrayAnnot.unique_chain_and_add_ids(atom_array)

    # get lig entity sequences and position
    label_entity_id_to_sequences = {}
    lig_chain_ids = []  # record chain_id of the first asym chain
    for label_entity_id in np.unique(atom_array.label_entity_id):
        if label_entity_id not in parser.entity_poly_type:
            current_lig_chain_ids = np.unique(
                atom_array.chain_id[atom_array.label_entity_id == label_entity_id]
            ).tolist()
            lig_chain_ids += current_lig_chain_ids
            for chain_id in current_lig_chain_ids:
                lig_atom_array = atom_array[atom_array.chain_id == chain_id]
                starts = get_residue_starts(lig_atom_array, add_exclusive_stop=True)
                seq = lig_atom_array.res_name[starts[:-1]].tolist()
                label_entity_id_to_sequences[label_entity_id] = seq

    # find polymer modifications
    entity_id_to_mod_list = {}
    for entity_id, res_names in parser.get_poly_res_names(atom_array).items():
        modifications_list = []
        for idx, res_name in enumerate(res_names):
            if res_name not in STD_RESIDUES:
                position = idx + 1
                modifications_list.append([position, f"CCD_{res_name}"])
        if modifications_list:
            entity_id_to_mod_list[entity_id] = modifications_list

    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=False)
    chain_starts_atom_array = atom_array[chain_starts]

    json_dict = {
        "sequences": [],
    }
    if assembly_id is not None:
        json_dict["assembly_id"] = assembly_id

    unique_label_entity_id = np.unique(atom_array.label_entity_id)
    label_entity_id_to_entity_id_in_json = {}
    chain_id_to_copy_id_dict = {}
    for idx, label_entity_id in enumerate(unique_label_entity_id):
        entity_id_in_json = str(idx + 1)
        label_entity_id_to_entity_id_in_json[label_entity_id] = entity_id_in_json
        chain_ids_in_entity = chain_starts_atom_array.chain_id[
            chain_starts_atom_array.label_entity_id == label_entity_id
        ]
        for chain_count, chain_id in enumerate(chain_ids_in_entity):
            chain_id_to_copy_id_dict[chain_id] = chain_count + 1
    copy_id = np.vectorize(chain_id_to_copy_id_dict.get)(atom_array.chain_id)
    atom_array.set_annotation("copy_id", copy_id)

    all_entity_counts = {}
    skipped_entity_id = []
    included_label_entity_ids = []
    for label_entity_id in unique_label_entity_id:
        print('label entity_id', label_entity_id)
        entity_dict = {}
        asym_chains = chain_starts_atom_array[
            chain_starts_atom_array.label_entity_id == label_entity_id
        ]
        entity_type = parser.entity_poly_type.get(label_entity_id, "ligand")
        if entity_type != "ligand":
            if entity_type == "polypeptide(L)":
                entity_type = "proteinChain"
            elif entity_type == "polydeoxyribonucleotide":
                entity_type = "dnaSequence"
            elif entity_type == "polyribonucleotide":
                entity_type = "rnaSequence"
            else:
                # DNA/RNA hybrid, polypeptide(D), etc.
                skipped_entity_id.append(label_entity_id)
                continue

            sequence = entity_seq.get(label_entity_id)
            entity_dict["sequence"] = sequence
        else:
            # ligand
            lig_ccd = "_".join(label_entity_id_to_sequences[label_entity_id])
            entity_dict["ligand"] = f"CCD_{lig_ccd}"
        entity_dict["count"] = len(asym_chains)
        all_entity_counts[label_entity_id_to_entity_id_in_json[label_entity_id]] = len(
            asym_chains
        )
        if save_entity_and_asym_id:
            entity_dict["label_entity_id"] = str(label_entity_id)
            entity_dict["label_asym_id"] = asym_chains.label_asym_id.tolist()

        # add PTM info
        if label_entity_id in entity_id_to_mod_list:
            modifications = entity_id_to_mod_list[label_entity_id]
            if entity_type == "proteinChain":
                entity_dict["modifications"] = [
                    {"ptmPosition": position, "ptmType": mod_ccd_code}
                    for position, mod_ccd_code in modifications
                ]
            else:
                entity_dict["modifications"] = [
                    {"basePosition": position, "modificationType": mod_ccd_code}
                    for position, mod_ccd_code in modifications
                ]
        json_dict["sequences"].append({entity_type: entity_dict})
        included_label_entity_ids.append(label_entity_id)
    # skip some uncommon entities
    atom_array = atom_array[~np.isin(atom_array.label_entity_id, skipped_entity_id)]

    # add covalent bonds
    atom_array = AddAtomArrayAnnot.add_token_mol_type(
        atom_array, parser.entity_poly_type
    )
    lig_polymer_bonds = get_ligand_polymer_bond_mask(atom_array, lig_include_ions=False)
    lig_lig_bonds = get_lig_lig_bonds(atom_array, lig_include_ions=False)
    inter_entity_bonds = np.vstack((lig_polymer_bonds, lig_lig_bonds))

    lig_indices = np.where(np.isin(atom_array.chain_id, lig_chain_ids))[0]
    lig_bond_mask = np.any(np.isin(inter_entity_bonds[:, :2], lig_indices), axis=1)
    inter_entity_bonds = inter_entity_bonds[lig_bond_mask]  # select bonds of ligands
    if inter_entity_bonds.size != 0:
        covalent_bonds = []
        for atoms in inter_entity_bonds[:, :2]:
            bond_dict = {}
            for i in range(2):
                atom = atom_array[atoms[i]]
                positon = atom.res_id
                bond_dict[f"entity{i+1}"] = int(
                    label_entity_id_to_entity_id_in_json[atom.label_entity_id]
                )
                bond_dict[f"position{i+1}"] = int(positon)
                bond_dict[f"atom{i+1}"] = atom.atom_name
                bond_dict[f"copy{i+1}"] = int(atom.copy_id)

            covalent_bonds.append(bond_dict)
    ordered_indices = []
    entity_ids = []
    atom_names = []
    entity_index = 0
    for label_entity_id in included_label_entity_ids:
        chain_ids = chain_starts_atom_array.chain_id[
            chain_starts_atom_array.label_entity_id == label_entity_id
            ]
        for chain_id in chain_ids:
            chain_indices = np.where(atom_array.chain_id == chain_id)[0]
            ordered_indices.extend(chain_indices)
            entity_ids.extend([entity_index] * len(chain_indices))
            entity_index += 1

    gt_coordinates = atom_array.coord[ordered_indices]
    atom_names = atom_array.atom_name[ordered_indices]

    return gt_coordinates, entity_ids, atom_names, atom_array[ordered_indices]
def cif_to_input_json(
    mmcif_file: str,
    assembly_id: str = None,
    altloc="first",
    output_json: str = None,
    sample_name=None,
    save_entity_and_asym_id=False,
) -> Tuple[list, list, list, list]:
    """
    Convert mmcif file to Protenix input json file.

    Args:
        mmcif_file (str): mmCIF file path.
        assembly_id (str, optional): Assembly ID. Defaults to None.
        altloc (str, optional): Altloc selection. Defaults to "first".
        output_json (str, optional): Output json file path. Defaults to None.
        sample_name (_type_, optional): The "name" filed in json file. Defaults to None.
        save_entity_and_asym_id (bool, optional): Whether to save entity and asym ids to json.
                                                  Defaults to False.

    Returns:
        dict: Protenix input json dict.
    """
    parser = MMCIFParser(mmcif_file)
    atom_array = gen_a_bioassembly_data_dict(mmcif_file, None, None, True)['atom_array']
    if any(["DIFFRACTION" in m for m in parser.methods]):
        atom_array = Filter.remove_crystallization_aids(
            atom_array, parser.entity_poly_type
        )
    gt_coordinates, entity_ids, atom_names, atom_array = atom_array_to_input_json(atom_array,
        parser,
        assembly_id,
        output_json,
        sample_name,
        save_entity_and_asym_id=save_entity_and_asym_id)
    return gt_coordinates, entity_ids, atom_names, atom_array

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
        # self.model = Protenix(self.configs).to(self.device)
        self.model = ProtenixSelect(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        # checkpoint_path = (
        #     f"{self.configs.load_checkpoint_dir}/{self.configs.model_name}.pt"
        # )
        checkpoint_path = 'output/consistency/17999_ema_0.999.pt'
        print('checkpoint_path', checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
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
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
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
        pdb_id = data['sample_name']
        gt_coordinates, entity_ids, atom_names, atom_array = cif_to_input_json(
            # f'/home/u3590540/cxlu/proteinx_data/mmcif_sub/{pdb_id}.cif')
        # f'/home/lab_JiangHB/cxlu/data/proteinx_data/longer-groundtruth/{str(pdb_id).upper()}.cif')
        f'/home/cxlu/data/proteinx_data/mmcif/{pdb_id}.cif')
        # f'/home/lab_JiangHB/cxlu/data/proteinx_data/mmcif_test/{pdb_id}.cif')
        # f'/home/u3590540/cxlu/proteinx_data/posebusters_mmcif/{pdb_id}.cif')
        gt_coordinates = torch.tensor(gt_coordinates)



        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                gt_coordinates=gt_coordinates,
                is_resolved=atom_array.is_resolved,
                pdb_id=pdb_id
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
                prediction = runner.predict(data)
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
