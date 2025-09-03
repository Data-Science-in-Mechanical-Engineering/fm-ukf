from pathlib import Path
import h5py

def merge_h5_files(source_dir: str, dest_file: str):
    """Simple H5 file merger"""
    try:
        import h5py
        
        source_path = Path(source_dir)
        dest_path = Path(dest_file)
        
        # Find all rank_*.h5 files
        rank_files = list(source_path.glob("rank_*.h5"))
        
        if not rank_files:
            print(f"No rank_*.h5 files found in {source_dir}")
            return
        
        print(f"Found {len(rank_files)} files to merge")
        
        # Create destination file
        with h5py.File(dest_path, 'w') as dest_f:
            # Initialize groups
            dest_f.create_group("True")
            dest_f.create_group("Sensors") 
            dest_f.create_group("Estimates")
            
            # Merge each rank file
            for rank_file in rank_files:
                print(f"  Merging {rank_file.name}...")
                
                with h5py.File(rank_file, 'r') as src_f:
                    # Copy all datasets from source to destination
                    def copy_datasets(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            if name not in dest_f:
                                dest_f.create_dataset(name, data=obj[:])
                            else:
                                print(f"    Skipping duplicate dataset: {name}")
                    
                    src_f.visititems(copy_datasets)
        
        print(f"Merge complete: {dest_path}")
            
    except Exception as e:
        print(f"Merge failed: {e}")

