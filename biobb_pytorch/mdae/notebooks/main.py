import os, sys, json, argparse
project_root = "/gpfs/projects/bsc23/MN4/bsc23/bsc23645/AI_ML/biobb_pytorch/"
sys.path.append(project_root)
from biobb_pytorch.mdae.mdfeaturizer import MDFeaturizer
from biobb_pytorch.mdae.build_model import buildModel
from biobb_pytorch.mdae.train_model import trainModel
from biobb_pytorch.mdae.evaluate_model import evaluateModel
from biobb_pytorch.mdae.encode_model import evaluateEncoder
from biobb_pytorch.mdae.decode_model import evaluateDecoder
from biobb_pytorch.mdae.LRP import relevancePropagation
from biobb_pytorch.mdae.make_plumed import generatePlumed


def load_json(file_path):
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        sys.exit(1)

def check_file_exists(file_path, file_description):
    """Check if a file exists, raise FileNotFoundError if not."""
    if file_path and not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: {file_description} not found: {file_path}")

def main():
    # Set up ArgumentParser
    parser = argparse.ArgumentParser(description="ANM Analysis Pipeline with BioBB PyTorch")
    
    # MDFeaturizer arguments
    parser.add_argument('-f', '--input_trajectory_file', default=True, help='Input trajectory file (e.g., DCD)')
    parser.add_argument('-s', '--input_topology_file', default=True, help='Input topology file (e.g., PDB)')
    parser.add_argument('-l', '--input_labels_file', default=None, help='Input labels file (NPY, optional)')
    parser.add_argument('-w', '--input_weights_file', default=None, help='Input weights file (NPY, optional)')

    parser.add_argument('-d', '--dataset_pt_path', default='output.dcd.pt', help='Output dataset PT file')
    parser.add_argument('-i', '--stats_pt_path', default='output.dcd.stats.pt', help='Output stats PT file')
    parser.add_argument('-fp', '--featurizer_properties', default=None, help='JSON file with MDFeaturizer properties')

    # buildModel arguments
    parser.add_argument('-m', '--model_pth_file', default='model.pth', help='Output model PTH file for buildModel')
    parser.add_argument('-bp', '--build_properties', default=None, help='JSON file with buildModel properties')

    # trainModel arguments
    parser.add_argument('-t', '--model_metrics_file', default='model_metrics.npz', help='Output metrics NPZ file')
    parser.add_argument('-tp', '--train_properties', default=None, help='JSON file with trainModel properties')

    # evaluateModel arguments
    parser.add_argument('-em', '--eval_model_pth_path', default='model.pth', help='Input model PTH file for evaluateModel')
    parser.add_argument('-ed', '--eval_dataset_pt_path', default='output.dcd.pt', help='Input dataset PT file for evaluateModel')
    parser.add_argument('-mr', '--model_results_file', default='model_results.npz', help='Output results NPZ file')
    parser.add_argument('-ep', '--eval_properties', default=None, help='JSON file with evaluateModel properties')

    # relevancePropagation arguments
    parser.add_argument('-lrp', '--output_lrp_results_npz_path', default='lrp_results.npz', help='Output LRP results NPZ file (first frames)')
    parser.add_argument('-lrpp', '--lrp_properties', default=None, help='JSON file with LRP properties for first frames')

    # generatePlumed arguments
    parser.add_argument('-n', '--ndx_file', default=None, help='Input index file (NDX, optional)')
    parser.add_argument('-r', '--reference_pdb_file', default=None, help='Reference PDB file (optional)')
    parser.add_argument('-ptc', '--output_model_ptc_file', default='test_model.ptc', help='Output model PTC file')
    parser.add_argument('-plumed', '--output_plumed_file', default='plumed.dat', help='Output PLUMED DAT file')
    parser.add_argument('-pf', '--output_plumed_feature_file', default='features.dat', help='Output PLUMED feature DAT file')
    parser.add_argument('-pp', '--plumed_properties', default=None, help='JSON file with generatePlumed properties')

    args = parser.parse_args()

    # Step 1: MDFeaturizer
    if args.featurizer_properties:
        # check input files
        try:
            check_file_exists(args.input_trajectory_file, "Input trajectory file")
            check_file_exists(args.input_topology_file, "Input topology file")
            # Check optional input files (only if provided)
            if args.input_labels_file:
                check_file_exists(args.input_labels_file, "Input labels file")
            if args.input_weights_file:
                check_file_exists(args.input_weights_file, "Input weights file")
        except FileNotFoundError as e:
            print(e)
            return

        # Load JSON properties
        try:
            featurizer_properties = load_json(args.featurizer_properties)
        except Exception:
            return  # Error already printed in load_json

        # Run MDFeaturizer
        try:
            MDFeaturizer(
                input_trajectory_path=args.input_trajectory_file,
                input_topology_path=args.input_topology_file,
                input_labels_npy_path=args.input_labels_file,
                input_weights_npy_path=args.input_weights_file,
                output_dataset_pt_path=args.dataset_pt_path,
                output_stats_pt_path=args.stats_pt_path,
                properties=featurizer_properties
            )
            print(f"MDFeaturizer completed successfully. Outputs: {args.dataset_pt_path}, {args.stats_pt_path}")
        except Exception as e:
            print(f"Error during MDFeaturizer execution: {e}")
            return
    else:
        print("Skipping MDFeaturizer step due to missing properties file.")

    # Step 2: buildModel
    if build_properties:
        try:
            check_file_exists(args.stats_pt_path, "Input stats PT file")
            check_file_exists(args.model_pth_file, "Output model PTH file path")
        except FileNotFoundError as e:
            print(e)
            return
        
        # Load JSON properties
        try:
            build_properties = load_json(args.build_properties)
        except Exception as e:
            print(f"Error loading build properties: {e}")
            return

        try:
            buildModel(
                input_stats_pt_path=args.stats_pt_path,
                output_model_pth_path=args.model_pth_file,
                properties=build_properties
            )
        except Exception as e:
            print(f"Error during buildModel execution: {e}")
            return
    else:
        print("Skipping buildModel step due to missing input files or parameters.")

    # Step 3: trainModel
    if args.train_properties:
        # check input files
        try:
            check_file_exists(args.model_pth_file, "Input model PTH file")
            check_file_exists(args.dataset_pt_path, "Input dataset PT file")
            check_file_exists(args.model_metrics_file, "Input model metrics file")
        except FileNotFoundError as e:
            print(e)
            return

        try:
            train_properties = load_json(args.train_properties)
        except Exception as e:
            print(f"Error loading train properties: {e}")
            return

        try:
            trainModel(
                input_model_pth_path=args.model_pth_file,
                input_dataset_pt_path=args.dataset_pt_path,
                output_model_pth_path=args.model_pth_file,
                output_metrics_npz_path=args.model_metrics_file,
                properties=train_properties
            )
        except Exception as e:
            print(f"Error during trainModel execution: {e}")
            return
    else:
        print("Skipping trainModel step due to missing input files or parameters.")
    
    # Step 4: evaluateModel
    if args.model_results_file:
        # check input files
        try:
            check_file_exists(args.eval_model_pth_path, "Input model PTH file")
            check_file_exists(args.eval_dataset_pt_path, "Input dataset PT file")
        except FileNotFoundError as e:
            print(e)
            return
        
        # Load JSON properties
        try:
            eval_properties = load_json(args.eval_properties)
        except Exception as e:
            print(f"Error loading evaluateModel properties: {e}")
            return

        evaluateModel(
            input_model_pth_path=args.eval_model_pth_path,
            input_dataset_pt_path=args.eval_dataset_pt_path,
            output_results_npz_path=args.model_results_file,
            properties=eval_properties
        )

    # Step 5: relevancePropagation
    if lrp_properties:
        # check input files
        try:
            check_file_exists(args.eval_model_pth_path, "Input model PTH file")
            check_file_exists(args.eval_dataset_pt_path, "Input dataset PT file")
        except FileNotFoundError as e:
            print(e)
            return

        # Load JSON properties
        try:
            lrp_properties = load_json(args.lrp_properties)
        except Exception as e:
            print(f"Error loading LRP properties: {e}")
            return
        try:
            relevancePropagation(
                input_model_pth_path=args.eval_model_pth_path,
                input_dataset_pt_path=args.eval_dataset_pt_path,
                output_results_npz_path=args.output_lrp_results_npz_path,
                properties=lrp_properties
            )
        except Exception as e:
            print(f"Error during relevancePropagation execution: {e}")
            return

    # Step 6: generatePlumed
    if plumed_properties:
        # check input files
        try:
            check_file_exists(args.model_pth_file, "Input model PTH file")
            check_file_exists(args.stats_pt_path, "Input stats PT file")
            if args.ndx_file:
                check_file_exists(args.ndx_file, "Input NDX file")
            if args.reference_pdb_file:
                check_file_exists(args.reference_pdb_file, "Input reference PDB file")
        except FileNotFoundError as e:
            print(e)
            return
        
        # Load JSON properties
        try:
            plumed_properties = load_json(args.plumed_properties)
        except Exception as e:
            print(f"Error loading generatePlumed properties: {e}")
            return
        
        try:
            generatePlumed(
                input_model_pth_path=args.model_pth_file,
                input_stats_pt_path=args.stats_pt_file,
                output_plumed_file=args.output_plumed_file,
                output_plumed_feature_file=args.output_plumed_feature_file,
            properties=plumed_properties
            )
        except Exception as e:
            print(f"Error during generatePlumed execution: {e}")
            return


    # Step 7: evaluateEncoder
    if enc_properties:
        # check input files
        try:
            check_file_exists(args.enc_input_model_pth_path, "Input model PTH file")
            check_file_exists(args.enc_input_dataset_pt_path, "Input dataset PT file")
        except FileNotFoundError as e:
            print(e)
            return

        # Load JSON properties
        try:
            enc_properties = load_json(args.enc_properties)
        except Exception as e:
            print(f"Error loading evaluateEncoder properties: {e}")
            return

        evaluateEncoder(
            input_model_pth_path=args.enc_input_model_pth_path,
            input_dataset_pt_path=args.enc_input_dataset_pt_path,
            output_results_npz_path=args.output_encoder_results_file,
            properties=enc_properties
        )

    # Step 8: evaluateDecoder
    if dec_properties:
        # check input files
        try:
            check_file_exists(args.dec_input_model_pth_path, "Input model PTH file")
            check_file_exists(args.input_dataset_npy_path, "Input dataset NPY file")
        except FileNotFoundError as e:
            print(e)
            return
        
        # Load JSON properties
        try:
            dec_properties = load_json(args.dec_properties)
        except Exception as e:
            print(f"Error loading evaluateDecoder properties: {e}")
            return  
        
        try:
            evaluateDecoder(
                input_model_pth_path=args.dec_input_model_pth_path,
                input_dataset_npy_path=args.input_dataset_npy_path,
                output_results_npz_path=args.output_decoded_results_npz_path,
                properties=dec_properties
            )
        except Exception as e:
            print(f"Error during evaluateDecoder execution: {e}")
            return
    

if __name__ == '__main__':
    main()
