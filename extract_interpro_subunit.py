import json
import os
import subprocess
import glob
from pathlib import Path


def write_fasta(sequence, output_file, header=">Protein_sequence"):
    """
    Writes a protein sequence to a FASTA file.

    Args:
        sequence (str): The protein sequence (e.g., "MKTIIALSYIFCLVFADYKDDDDK").
        output_file (str): The name of the output FASTA file.
        header (str): The header for the FASTA file (default is ">Protein_sequence").
    """
    # Ensure the sequence is uppercase and remove any whitespace
    sequence = sequence.strip().upper()

    with open(output_file, "w") as file:
        # Write the header
        file.write(header + "\n")

        # Write the sequence in 60-character chunks (FASTA format convention)
        for i in range(0, len(sequence), 60):
            file.write(sequence[i:i + 60] + "\n")


def run_interproscan(fasta_file, datadir="/data2/cxlu"):
    """
    Runs the InterProScan command for a given FASTA file.

    Args:
        fasta_file (str): Path to the FASTA file.
        datadir (str): Path to the data directory for InterProScan.

    Returns:
        int: Return code of the command (0 for success).
    """
    command = [
        "nextflow", "run", "ebi-pf-team/interproscan6",
        "-r", "6.0.0-beta",
        "-profile", "docker",
        "--datadir", datadir,
        "--input", fasta_file,
        "--applications", "CATH-Gene3D"
    ]

    print(f"\n{'=' * 80}")
    print(f"Running InterProScan for: {fasta_file}")
    print(f"Command: {' '.join(command)}")
    print(f"{'=' * 80}\n")

    try:
        # Run the command and wait for it to complete
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✓ Successfully completed InterProScan for {fasta_file}")
        print(result.stdout)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running InterProScan for {fasta_file}")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return e.returncode


def process_json_files(folder_path, datadir="/data2/cxlu", output_dir="./fasta_output"):
    """
    Processes all JSON files in a folder, extracts sequences, creates FASTA files,
    and runs InterProScan for each sequence.

    Args:
        folder_path (str): Path to the folder containing JSON files.
        datadir (str): Path to the data directory for InterProScan.
        output_dir (str): Directory to save FASTA files.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process\n")

    # Process each JSON file
    for json_file in json_files:
        print(f"\n{'#' * 80}")
        print(f"Processing JSON file: {json_file}")
        print(f"{'#' * 80}\n")

        try:
            # Read the JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Process each entry in the JSON (usually a list with one item)
            for entry_idx, entry in enumerate(data):
                name = entry.get('name', f'unknown_{entry_idx}')
                sequences = entry.get('sequences', [])

                print(f"Entry name: {name}")
                print(f"Number of sequences: {len(sequences)}\n")

                # Process each sequence
                for seq_idx, seq_entry in enumerate(sequences):
                    if 'proteinChain' in seq_entry:
                        protein_chain = seq_entry['proteinChain']
                        sequence = protein_chain.get('sequence', '')

                        if sequence:
                            # Create FASTA filename
                            fasta_filename = f"{name}_seq_{seq_idx}.fasta"
                            fasta_path = os.path.join(output_dir, fasta_filename)

                            # Create header with more information
                            header = f">{name}_chain_{seq_idx}"

                            # Write FASTA file
                            print(f"Creating FASTA file: {fasta_path}")
                            print(f"Sequence length: {len(sequence)} amino acids")
                            write_fasta(sequence, fasta_path, header)

                            # Run InterProScan
                            run_interproscan(fasta_path, datadir)
                        else:
                            print(f"Warning: Empty sequence found for {name}, sequence {seq_idx}")
                    else:
                        print(f"Warning: No 'proteinChain' found in sequence {seq_idx}")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {json_file}: {e}")
        except Exception as e:
            print(f"Error processing JSON file {json_file}: {e}")

    print(f"\n{'#' * 80}")
    print("All JSON files processed!")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    # Configuration
    # FOLDER_PATH = "/home/cxlu/protein/Protenix/Pre_process/output_hpc_recentpdb/4"  # Change this to your folder containing JSON files
    FOLDER_PATH = "/home/cxlu/protein/Protenix/Pre_process/longer-raw-data-cryo/1"  # Change this to your folder containing JSON files
    DATADIR = "/data2/cxlu"  # Change this if needed
    OUTPUT_DIR = "./fasta_output"  # Directory to save FASTA files

    # Process all JSON files
    process_json_files(FOLDER_PATH, DATADIR, OUTPUT_DIR)
