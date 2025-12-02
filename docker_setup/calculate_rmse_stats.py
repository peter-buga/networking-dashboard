import os
import json
import glob
import statistics

def calculate_rmse_stats(models_dir):
    print(f"Searching for models in: {models_dir}")
    pattern = os.path.join(models_dir, "**", "latest.json")
    model_files = glob.glob(pattern, recursive=True)
    
    rmse_values = []
    packet_rmse_values = []
    octet_rmse_values = []
    zero_rmse_count = 0
    
    print(f"Found {len(model_files)} model files.")
    print("-" * 80)
    print(f"{'Metric':<30} | {'Hostname':<15} | {'Interface':<15} | {'RMSE':<10}")
    print("-" * 80)

    for file_path in model_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            rmse = data.get('rmse')
            metric = data.get('metric', 'unknown')
            labels = data.get('labels', {})
            hostname = labels.get('hostname', 'N/A')
            interface = labels.get('interface_name', labels.get('stream_name', 'N/A'))
            
            if rmse is not None:
                if rmse > 0.0001: # Filter out effectively zero values
                    rmse_values.append(rmse)
                    if 'packets' in metric:
                        packet_rmse_values.append(rmse)
                    elif 'octets' in metric:
                        octet_rmse_values.append(rmse)
                    
                    print(f"{metric:<30} | {hostname:<15} | {interface:<15} | {rmse:<10.4f}")
                else:
                    zero_rmse_count += 1

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print("-" * 80)
    print(f"Total models processed: {len(model_files)}")
    print(f"Models with 0 RMSE (ignored): {zero_rmse_count}")
    print(f"Models with >0 RMSE: {len(rmse_values)}")
    
    if rmse_values:
        avg_rmse = sum(rmse_values) / len(rmse_values)
        min_rmse = min(rmse_values)
        max_rmse = max(rmse_values)
        median_rmse = statistics.median(rmse_values)
        
        print("-" * 80)
        print(f"Overall Average RMSE: {avg_rmse:.4f}")
        print(f"Overall Median RMSE:  {median_rmse:.4f}")
        print(f"Overall Min RMSE:     {min_rmse:.4f}")
        print(f"Overall Max RMSE:     {max_rmse:.4f}")
        
        if packet_rmse_values:
            avg_packet_rmse = sum(packet_rmse_values) / len(packet_rmse_values)
            print("-" * 40)
            print(f"Packets Average RMSE: {avg_packet_rmse:.4f}")
            print(f"Packets Count:        {len(packet_rmse_values)}")

        if octet_rmse_values:
            avg_octet_rmse = sum(octet_rmse_values) / len(octet_rmse_values)
            print("-" * 40)
            print(f"Octets Average RMSE:  {avg_octet_rmse:.4f}")
            print(f"Octets Count:         {len(octet_rmse_values)}")

    else:
        print("No non-zero RMSE values found.")

if __name__ == "__main__":
    # Adjust path as needed based on where the script is run
    models_directory = "/home/peti/networking-dashboard/docker_setup/forecaster_models"
         
    calculate_rmse_stats(models_directory)
