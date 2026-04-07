import os
import random

from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

from Bio.PDB import MMCIFParser
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import shutil
def is_protein_complex(cif_file):
    """
    Determines if a given CIF file represents a protein monomer or complex.
    Ignores DNA, RNA, and ligands.

    Args:
        cif_file (str): Path to the CIF file.

    Returns:
        str: 'Monomer' if the structure is a monomer, 'Complex' if it is a complex.
    """
    parser = MMCIFParser(QUIET=True)
    residue_count = 0
    try:
        # Parse the CIF file
        structure = parser.get_structure('structure', cif_file)

        # Get all chains that are proteins
        protein_chains = []
        for model in structure:
            for chain in model:
                residue_count += len(chain)
                for residue in chain:
                    if residue.id[0] == " ":  # Standard residues (ignore heteroatoms)
                        protein_chains.append(chain.id)
                        break  # Stop checking residues in this chain

        # Get unique chain IDs
        unique_protein_chains = set(protein_chains)

        # Determine if it's a monomer or a complex
        if len(unique_protein_chains) == 1:
            return False, residue_count
        elif len(unique_protein_chains) > 1:
            return True, residue_count
        else:
            return False, residue_count

    except Exception as e:
        return False, residue_count

def protein_complex_list():
    protein_pdb_list = []
    for file in os.listdir('/home/cxlu/protein/Protenix/Pre_process/output_recentpdb_only_protein'):
        if file.endswith('.json'):
            pdb_id = file.split('-')[0]
            protein_pdb_list.append(pdb_id)
    return protein_pdb_list

# prediction_path = './recentpdb_pae_subunit_wholechainwrite_split10_dist20_pae5_0.5percentile_isresolvedfixed'
prediction_path = './recentpdb_interpro_subunit_nearest_5dist_asymid_nomax_iptm_3pae_v0.7_no20_max1000'
# prediction_path = './recentpdb_pae_subunit_wholechainwrite_split10_dist20_pae5_no20short'
prediction_path2 = '../../Protenix4/Protenix/recentpdb_prediction'
# prediction_path2 = './recentpdb_interpro_subunit'
# prediction_path2 = '/data2/cxlu/protenix_data/evaluation/inference_output/protenix/RecentPDB'

gt_path = '/home/cxlu/data/proteinx_data/mmcif/'
# gt_path = '/data/Protein_baseline/AlphaFold-Multimer/longer-groundtruth/'
# gt_path = './mmcif/'
all_success = 0
all_success_af3 = 0
all_fail = 0
all_fail_af3 = 0
if os.path.exists("ignored_list.pkl"):
    with open('ignored_list.pkl', 'rb') as f:
        ignore_pdb = pickle.load(f)
else:
    ignore_pdb = [] # too long time inference
print('ignore list', ignore_pdb)
# fail_case = ['8dao', '8a1e', '8dwc', '8dtx', '7um3', '7upq', '7u0e', '8a1e', '8dtt', '7rew', '8dtr', '7t82', '7vgr', '7vge'] # 7tud, 7qs8比较奇怪，af3 sucess为4，fail为0， 但是AF3-large success fail都为0
fail_case = [] # 7tud, 7qs8比较奇怪，af3 sucess为4，fail为0， 但是AF3-large success fail都为0
failed_case_af3 = ['7vcq', '8f0a', '7vdd', '7pb2', '8d6z', '7xy8', '7tz5', '8gpt', '7sts', '7ppc', '7w71', ] # AF3严重失败
fail_compensate = 0
# protein_pdb_list = protein_complex_list()
# protein_pdb_list = ['7n1m', '7oit', '7npo', '7qre']
has_meet_last_ignore = False
record = {}

def evaluation (pdb_id, pred_path):
    if pdb_id == 'ERR' or pdb_id in ignore_pdb:
        return None, 0, 0, pdb_id
    # if not has_meet_last_ignore:
    #     continue
    # if pdb_id != '7v9u':
    #     continue

    best_success = 0
    best_fail = 0
    best_scores = []
    seeds = [101]
    for i in range(5):
        sucess_count = 0
        fail_count = 0
        for seed in seeds:
            prediction_file = f'{pred_path}/{pdb_id}/seed_{seed}/predictions/{pdb_id}_sample_{i}.cif'
            if not os.path.exists(prediction_file):
                prediction_file = f'{pred_path}/{pdb_id}/seed_{seed}/predictions/{pdb_id}_seed_{seed}_sample_{i}.cif'

        #     prediction_file = f'{pred_path}/{pdb_id}/{pdb_id}/seed_{seed}/predictions/{pdb_id}_sample_{i}.cif'
            gt_file = f'{gt_path}/{pdb_id}.cif'
            # print('prediction path', prediction_file)
            # print('gt file', gt_file)
            # is_complex, residue_count = is_protein_complex(gt_file)
            # if is_complex:
            cmd = ['DockQ', prediction_file,  gt_file, '--short']
            # print(cmd)
            try:
                dockq_score = subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True, timeout=60)
            except subprocess.TimeoutExpired as e:
                return False, 0, 0, pdb_id
            # if dockq_score.returncode == 0:
            result = dockq_score.stdout
            # print('result', result)
            if i == 0 and seed == 101:
                num_interface = 0
                for line in result.split('\n'):
                    if line.startswith('DockQ'):
                        num_interface += 1
                best_scores = [0. for _ in range(num_interface)]
            for interface_id, line in enumerate(result.split('\n')[1:]):
                
                if line.startswith('DockQ'):
                    score = float(line.split(' ')[1])
                    best_scores[interface_id - 1] = max(best_scores[interface_id - 1], score)
            # print('dockq score:', dockq_score.stdout)
            for score in best_scores:
                if score > 0.23:
                    sucess_count += 1
                else:
                    fail_count += 1
            if sucess_count >= best_success:
                best_fail = fail_count
                best_success = sucess_count
            if fail_count == 0:
                break
    # print(best_success, best_fail, pdb_id)
    return True, best_success, best_fail, pdb_id

def evaluation_compare(pdb_id):

    status1, best_success_af3, best_fail_af3, _ = evaluation(pdb_id, prediction_path2)
    status2, best_success, best_fail, _ = evaluation(pdb_id, prediction_path)
    if not status1 or not status2:
        return False, 0, 0, 0, 0, pdb_id
    if best_success != best_success_af3:
        print('different', pdb_id, best_success, best_success_af3, best_fail, best_fail_af3)
    else:
        print('same', pdb_id, best_success, best_success_af3, best_fail, best_fail_af3)
    return True, best_success, best_fail, best_success_af3, best_fail_af3, pdb_id

pdb_id_list = []
for pdb_id in os.listdir(prediction_path):
    if pdb_id == 'ERR' or pdb_id in ignore_pdb or pdb_id not in os.listdir(prediction_path2) or pdb_id in fail_case or pdb_id in failed_case_af3:
        if pdb_id not in os.listdir(prediction_path2):
            print(pdb_id, 'not have prediction')
            # from glob import glob
            # pred_files = glob('/home/cxlu/protein/Protenix/Pre_process/output_hpc_recentpdb/*/'+pdb_id+'*')
            # if len(pred_files) != 0:
            #     if random.random() < 0.5:
            #         shutil.copy(pred_files[0], f'/home/cxlu/protein/Protenix/Pre_process/output_hpc_recentpdb_miss/1/{os.path.basename(pred_files[0])}')
            #     else:
            #         shutil.copy(pred_files[0], f'/home/cxlu/protein/Protenix/Pre_process/output_hpc_recentpdb_miss/2/{os.path.basename(pred_files[0])}')
            # else:
            #     print('also not have in recentpdb', pdb_id)
        continue
    pdb_id_list.append(pdb_id)
print(pdb_id_list)
all_failed_pdb_list = ignore_pdb
with Pool(20) as pool:
    for status, success, fail,success_af3, fail_af3, pdb_id in tqdm(pool.imap(evaluation_compare, pdb_id_list)):
        if status is not None:
            if status:
                all_success += success
                all_success_af3 += success_af3
                all_fail += fail
                all_fail_af3 += fail_af3
                record[pdb_id] = {'success': success, 'fail': fail}
            else:
                all_failed_pdb_list.append(pdb_id)
            # print(f'{pdb_id} success: {success}, fail: {fail}')
print(prediction_path)
print(all_success, all_fail, all_success/(all_success+all_fail - fail_compensate))
print(all_success_af3, all_fail_af3, all_success_af3/(all_success_af3+all_fail_af3 - fail_compensate))

with open(f"ignored_list.pkl", 'wb') as f:
    pickle.dump(all_failed_pdb_list, f)